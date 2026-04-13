import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from app import config
from app.models.schemas import TaskInfo, TaskStatus

logger = logging.getLogger(__name__)

MAX_COMPLETED_AGE_SEC = 3600  # 완료 작업 1시간 후 자동 삭제
MAX_STORE_SIZE = 100          # 최대 보관 건수


class JobStore:
    """Thread-safe in-memory job state store with TTL cleanup."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        self._ensure_results_dir()

    def _ensure_results_dir(self) -> Path:
        """Ensure results directory exists. Use /dev/shm if available, fallback to /tmp."""
        target_dir = config.RESULTS_DIR
        if target_dir.parent.name == "shm":
            # /dev/shm case: check if /dev/shm exists
            shm_parent = Path("/dev/shm")
            if not shm_parent.exists():
                logger.warning("/dev/shm not available, falling back to /tmp")
                target_dir = Path("/tmp/stt-results")
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def _get_results_dir(self) -> Path:
        """Get the actual results directory (may be different from config if /dev/shm unavailable)."""
        if Path("/dev/shm").exists():
            return config.RESULTS_DIR
        return Path("/tmp/stt-results")

    def create(self, task_id: str) -> TaskInfo:
        task = TaskInfo(task_id=task_id, status=TaskStatus.pending)
        with self._lock:
            self._tasks[task_id] = task
            self._timestamps[task_id] = time.time()
            self._cleanup_expired()
        return task

    def get(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(task_id)

    def active_count(self) -> int:
        """pending + processing 상태인 작업 수 (큐 백프레셔용)."""
        with self._lock:
            return sum(
                1 for t in self._tasks.values()
                if t.status in (TaskStatus.pending, TaskStatus.processing)
            )

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                self._tasks[task_id] = task.model_copy(update={"status": status})

    def set_result(self, task_id: str, result: dict) -> None:
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                self._tasks[task_id] = task.model_copy(
                    update={"status": TaskStatus.completed, "result": result}
                )

    def set_error(self, task_id: str, error: str) -> None:
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                self._tasks[task_id] = task.model_copy(
                    update={"status": TaskStatus.failed, "error": error}
                )


    def set_audio(self, task_id: str, audio_files: dict[str, bytes]) -> None:
        """Write audio files to disk under per-task subdirectory."""
        results_dir = self._get_results_dir()
        task_audio_dir = results_dir / task_id
        task_audio_dir.mkdir(parents=True, exist_ok=True)

        for filename, wav_bytes in audio_files.items():
            audio_path = task_audio_dir / filename
            audio_path.write_bytes(wav_bytes)
            logger.info("[%s] 오디오 저장: %s (%d bytes)", task_id, filename, len(wav_bytes))

    def get_audio(self, task_id: str, filename: str) -> Path | None:
        """Return Path to audio file if it exists, None otherwise."""
        results_dir = self._get_results_dir()
        task_dir = results_dir / task_id
        audio_path = (task_dir / filename).resolve()
        # Path traversal 방지: 결과 경로가 task 디렉토리 내부인지 확인
        if not str(audio_path).startswith(str(task_dir.resolve())):
            logger.warning("[%s] Path traversal 시도 차단: %s", task_id, filename)
            return None
        if audio_path.exists() and audio_path.is_file():
            return audio_path
        logger.warning("[%s] Audio not found: %s", task_id, filename)
        return None


    def _cleanup_expired(self) -> None:
        """Remove completed/failed tasks older than MAX_COMPLETED_AGE_SEC. Must hold lock.

        Also marks pending/processing tasks as failed if they exceed MAX_PROCESSING_AGE_SEC
        to prevent stuck jobs from blocking the queue indefinitely.
        """
        now = time.time()
        expired = []
        for task_id, ts in self._timestamps.items():
            task = self._tasks.get(task_id)
            if task is None:
                expired.append(task_id)
                continue
            age = now - ts
            if task.status in (TaskStatus.pending, TaskStatus.processing):
                if age > config.MAX_PROCESSING_AGE_SEC:
                    logger.warning(
                        "[%s] Stuck job detected (status=%s, age=%.0fs) — marking failed",
                        task_id, task.status.value, age,
                    )
                    self._tasks[task_id] = task.model_copy(
                        update={"status": TaskStatus.failed, "error": "processing_timeout"}
                    )
            if age > MAX_COMPLETED_AGE_SEC and task.status in (TaskStatus.completed, TaskStatus.failed):
                expired.append(task_id)

        # Also evict oldest if over max size
        if len(self._tasks) > MAX_STORE_SIZE:
            by_age = sorted(self._timestamps.items(), key=lambda x: x[1])
            for task_id, _ in by_age[:len(self._tasks) - MAX_STORE_SIZE]:
                if task_id not in expired:
                    expired.append(task_id)

        results_dir = self._get_results_dir()
        for task_id in expired:
            self._tasks.pop(task_id, None)
            self._timestamps.pop(task_id, None)
            # Remove per-task audio directory
            task_audio_dir = results_dir / task_id
            if task_audio_dir.exists():
                shutil.rmtree(task_audio_dir, ignore_errors=True)
                logger.info("[%s] 오디오 디렉토리 삭제", task_id)

        if expired:
            logger.info("JobStore cleanup: %d건 제거 (남은: %d건)", len(expired), len(self._tasks))


job_store = JobStore()
