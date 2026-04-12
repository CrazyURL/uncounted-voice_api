from app.core.job_store import job_store, JobStore
from app.models.schemas import TaskInfo, TaskStatus


class TestJobStore:
    def test_create_task(self):
        store = JobStore()
        task = store.create("test-001")
        assert task.task_id == "test-001"
        assert task.status == TaskStatus.pending

    def test_set_result(self):
        store = JobStore()
        store.create("test-002")
        store.set_result("test-002", {"text": "hello"})
        task = store.get("test-002")
        assert task.status == TaskStatus.completed
        assert task.result == {"text": "hello"}

    def test_set_error(self):
        store = JobStore()
        store.create("test-003")
        store.set_error("test-003", "something failed")
        task = store.get("test-003")
        assert task.status == TaskStatus.failed
        assert task.error == "something failed"

    def test_get_nonexistent(self):
        store = JobStore()
        assert store.get("nope") is None
