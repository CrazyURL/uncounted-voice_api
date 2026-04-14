"""큐 백프레셔: MAX_ACTIVE_JOBS 초과 시 POST /transcribe가 503 반환."""
from app import config
from app.core.job_store import JobStore, job_store
from app.models.schemas import TaskStatus


class TestActiveCount:
    def test_pending_counts_as_active(self):
        store = JobStore()
        store.create("a")
        store.create("b")
        assert store.active_count() == 2

    def test_processing_counts_as_active(self):
        store = JobStore()
        store.create("a")
        store.update_status("a", TaskStatus.processing)
        assert store.active_count() == 1

    def test_completed_does_not_count(self):
        store = JobStore()
        store.create("a")
        store.set_result("a", {"text": "x"})
        assert store.active_count() == 0

    def test_failed_does_not_count(self):
        store = JobStore()
        store.create("a")
        store.set_error("a", "boom")
        assert store.active_count() == 0

    def test_mixed(self):
        store = JobStore()
        store.create("a")  # pending
        store.create("b")
        store.update_status("b", TaskStatus.processing)
        store.create("c")
        store.set_result("c", {})
        store.create("d")
        store.set_error("d", "err")
        assert store.active_count() == 2  # a, b only


class TestQueueDepthEndpoint:
    def test_queue_full_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 2)

        # 2개 pending 작업을 인위적으로 채움
        job_store.create("t1")
        job_store.create("t2")
        assert job_store.active_count() == 2

        # 3번째 POST는 503
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
        )

        assert response.status_code == 503
        body = response.json()
        assert body["detail"]["error"] == "queue_full"
        assert body["detail"]["active_jobs"] == 2
        assert body["detail"]["max_active_jobs"] == 2
        assert body["detail"]["retry_after_sec"] == config.QUEUE_FULL_RETRY_AFTER_SEC
        assert response.headers.get("Retry-After") == str(config.QUEUE_FULL_RETRY_AFTER_SEC)

    def test_under_limit_proceeds_past_queue_check(self, client, monkeypatch):
        """큐 여유 있으면 백프레셔 체크는 통과 — 확장자 검증 등 다음 단계로 진행."""
        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 5)

        # 잘못된 확장자로 POST → 503이 아닌 400이 나와야 한다 (큐 체크는 통과)
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.xyz", b"garbage", "application/octet-stream")},
        )

        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]

    def test_completed_tasks_do_not_block(self, client, monkeypatch):
        """완료/실패 작업은 백프레셔 카운트에 포함되지 않아야 한다."""
        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 2)

        # 2개 완료 + 2개 실패 = 모두 active가 아님
        for tid in ("c1", "c2"):
            job_store.create(tid)
            job_store.set_result(tid, {})
        for tid in ("f1", "f2"):
            job_store.create(tid)
            job_store.set_error(tid, "err")
        assert job_store.active_count() == 0

        # 새 요청은 확장자 검증 단계까지 진행 (400 반환)
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.xyz", b"garbage", "application/octet-stream")},
        )
        assert response.status_code == 400


class TestExtensionValidation:
    """POST /transcribe 확장자 allowlist 및 거절 로그 PII 안전성 검증."""

    def test_amr_extension_accepted(self, client, monkeypatch):
        """Android 통화 녹음 포맷 amr이 400으로 거절되지 않아야 함."""
        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 5)
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("call.amr", b"\x00" * 256, "audio/amr")},
        )
        assert response.status_code != 400
        if response.status_code == 200:
            assert "task_id" in response.json()

    def test_3gp_extension_accepted(self, client, monkeypatch):
        """Android 통화 녹음 포맷 3gp가 400으로 거절되지 않아야 함."""
        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 5)
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("call.3gp", b"\x00" * 256, "audio/3gpp")},
        )
        assert response.status_code != 400

    def test_reject_400_log_omits_filename(self, client, monkeypatch, caplog):
        """지원하지 않는 확장자 거절 시 warning 로그에 파일명(PII) 기록 금지."""
        import logging

        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 5)
        caplog.set_level(logging.WARNING, logger="app.routers.transcribe")

        sensitive_filename = "통화 녹음 01022654502_260401_172236.xyz"
        response = client.post(
            "/api/v1/transcribe",
            files={"file": (sensitive_filename, b"garbage", "application/octet-stream")},
        )

        assert response.status_code == 400
        reject_records = [r for r in caplog.records if "[reject-400]" in r.getMessage()]
        assert len(reject_records) >= 1, "reject-400 warning 로그가 기록되지 않음"
        for record in reject_records:
            msg = record.getMessage()
            assert sensitive_filename not in msg, f"파일명 PII가 로그에 노출됨: {msg}"
            assert "01022654502" not in msg, f"전화번호 PII가 로그에 노출됨: {msg}"
            assert "ext='xyz'" in msg, f"확장자는 기록되어야 함: {msg}"


class TestHealthQueueField:
    """GET /api/v1/health가 큐 상태를 포함하는지 검증."""

    def test_health_includes_queue_when_empty(self, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 5)
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert "queue" in body
        assert body["queue"]["active"] == 0
        assert body["queue"]["max_active"] == 5
        assert body["queue"]["utilization_pct"] == 0.0

    def test_health_reflects_active_jobs(self, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 5)
        # 3개 활성 주입
        job_store.create("h1")
        job_store.create("h2")
        job_store.create("h3")
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        queue = response.json()["queue"]
        assert queue["active"] == 3
        assert queue["max_active"] == 5
        assert queue["utilization_pct"] == 60.0

    def test_health_utilization_at_capacity(self, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 2)
        job_store.create("h1")
        job_store.create("h2")
        response = client.get("/api/v1/health")
        queue = response.json()["queue"]
        assert queue["active"] == 2
        assert queue["utilization_pct"] == 100.0

    def test_health_completed_tasks_excluded_from_queue(self, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_ACTIVE_JOBS", 5)
        job_store.create("h1")
        job_store.set_result("h1", {})
        job_store.create("h2")  # 이것만 active
        response = client.get("/api/v1/health")
        assert response.json()["queue"]["active"] == 1
