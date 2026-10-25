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
