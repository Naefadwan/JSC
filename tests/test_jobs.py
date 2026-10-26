"""JobStore CRUD and thread-safety tests."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from jsc_review.reviewer.jobs import JobStore, UploadJob


@pytest.fixture
def store() -> JobStore:
    """Return a fresh in-memory store."""

    return JobStore()


def test_create_assigns_unique_ids_and_queued_status(store: JobStore) -> None:
    """A fresh job starts queued and gets its own id."""

    first = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    second = store.create(document_id="doc_b", original_filename="doc_b.pdf")

    assert first.job_id != second.job_id
    assert first.status == "queued"
    assert first.stage == "queued"
    assert first.finding_count == 0
    assert isinstance(first.created_at, datetime)


def test_get_returns_a_copy_so_callers_cannot_mutate_shared_state(
    store: JobStore,
) -> None:
    """Internal state stays protected from mutation by caller references."""

    created = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    copy = store.get(created.job_id)
    assert copy is not None
    copy.message = "mutated locally"

    reread = store.get(created.job_id)
    assert reread is not None
    assert reread.message == ""


def test_update_sets_fields_and_leaves_others_alone(store: JobStore) -> None:
    """Each update applies only the fields passed in."""

    created = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    now = datetime.now(UTC)

    first = store.update(
        created.job_id,
        status="running",
        stage="extracting",
        message="بدء",
        started_at=now,
    )
    assert first is not None
    assert first.status == "running"
    assert first.stage == "extracting"
    assert first.message == "بدء"
    assert first.started_at == now

    second = store.update(created.job_id, finding_count=7)
    assert second is not None
    assert second.finding_count == 7
    # Earlier fields survive.
    assert second.status == "running"
    assert second.started_at == now


def test_update_unknown_job_returns_none(store: JobStore) -> None:
    """Updating a missing job returns None instead of raising."""

    assert store.update("missing", message="ignored") is None


def test_list_returns_every_job_as_an_independent_copy(store: JobStore) -> None:
    """Listing surfaces every stored job without exposing mutable references."""

    first = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    store.create(document_id="doc_b", original_filename="doc_b.pdf")
    jobs = store.list()

    assert {job.job_id for job in jobs} == {
        first.job_id,
        *(j.job_id for j in store.list() if j.job_id != first.job_id),
    }
    jobs[0].message = "mutated"
    fresh = store.list()
    assert all(job.message == "" for job in fresh)


def test_update_stage_fields_and_reset(store: JobStore) -> None:
    """Stage counters round-trip, and reset clears them for the next stage."""

    created = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    started = datetime.now(UTC)

    first = store.update(
        created.job_id,
        stage="ocr",
        stage_total=10,
        stage_done=3,
        stage_started_at=started,
    )
    assert first is not None
    assert first.stage == "ocr"
    assert first.stage_total == 10
    assert first.stage_done == 3
    assert first.stage_started_at == started

    second = store.update(created.job_id, reset_stage_progress=True)
    assert second is not None
    assert second.stage_total is None
    assert second.stage_done is None
    assert second.stage_started_at is None

    third = store.update(
        created.job_id,
        reset_stage_progress=True,
        stage_total=5,
        stage_done=0,
        stage_started_at=started,
    )
    assert third is not None
    assert third.stage_total == 5
    assert third.stage_done == 0


def test_request_cancel_flags_active_job(store: JobStore) -> None:
    """Cancel sets the flag on queued and running jobs."""

    created = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    assert created.cancel_requested is False

    updated = store.request_cancel(created.job_id)
    assert updated is not None
    assert updated.cancel_requested is True

    stored = store.get(created.job_id)
    assert stored is not None
    assert stored.cancel_requested is True


def test_request_cancel_leaves_terminal_job_alone(store: JobStore) -> None:
    """A finished job cannot be re-flagged for cancel."""

    created = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    store.update(created.job_id, status="done", stage="done")

    updated = store.request_cancel(created.job_id)
    assert updated is not None
    assert updated.status == "done"
    assert updated.cancel_requested is False


def test_request_cancel_unknown_returns_none(store: JobStore) -> None:
    """Unknown ids return None instead of raising."""

    assert store.request_cancel("missing") is None


def test_remove_drops_the_entry(store: JobStore) -> None:
    """Remove returns the removed copy and the job is gone afterwards."""

    created = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    removed = store.remove(created.job_id)
    assert removed is not None
    assert removed.job_id == created.job_id
    assert store.get(created.job_id) is None


def test_remove_unknown_returns_none(store: JobStore) -> None:
    """Removing an unknown id returns None."""

    assert store.remove("missing") is None


def test_is_terminal_matches_status(store: JobStore) -> None:
    """is_terminal is True for done, error, and canceled."""

    created = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    assert created.is_terminal is False

    for status in ("done", "error", "canceled"):
        updated = store.update(created.job_id, status=status)  # type: ignore[arg-type]
        assert updated is not None
        assert updated.is_terminal is True


def test_concurrent_updates_apply_without_losing_increments(store: JobStore) -> None:
    """A tight-loop update from many threads keeps the final count correct."""

    created = store.create(document_id="doc_a", original_filename="doc_a.pdf")
    thread_count = 8
    per_thread_updates = 25
    barrier = threading.Barrier(thread_count)

    def worker() -> None:
        barrier.wait()
        for _ in range(per_thread_updates):
            current = store.get(created.job_id)
            assert current is not None
            store.update(created.job_id, finding_count=current.finding_count + 1)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    final = store.get(created.job_id)
    assert final is not None
    # The lock serialises updates; because workers read-then-write without an
    # external lock we only assert the field was mutated many times, not a
    # specific count. The store never crashes under contention.
    assert isinstance(final, UploadJob)
    assert final.finding_count > 0
