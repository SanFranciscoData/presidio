"""Unit tests for the shared per-key build lock.

These exercise the serialization primitive that the build-from-scratch
environments (Docker, E2B) use so concurrent trials of the same task don't all
build the same named artifact at once.
"""

from __future__ import annotations

import asyncio

from presidio.environments.build_lock import KeyedBuildLock


def test_same_key_returns_same_lock():
    locks = KeyedBuildLock()
    assert locks("alpha") is locks("alpha")


def test_different_keys_return_distinct_locks():
    locks = KeyedBuildLock()
    assert locks("alpha") is not locks("beta")


def test_same_key_serializes_concurrent_sections():
    """Two coroutines contending on the same key never overlap in the critical
    section; a third on a different key runs concurrently."""
    locks = KeyedBuildLock()
    active = 0
    max_concurrent_same_key = 0

    async def critical(key: str, order: list[str], tag: str) -> None:
        nonlocal active, max_concurrent_same_key
        async with locks(key):
            active += 1
            if key == "same":
                max_concurrent_same_key = max(max_concurrent_same_key, active)
            order.append(tag)
            await asyncio.sleep(0)  # yield so a contender could interleave
            active -= 1

    async def run() -> list[str]:
        order: list[str] = []
        await asyncio.gather(
            critical("same", order, "a"),
            critical("same", order, "b"),
            critical("other", order, "c"),
        )
        return order

    order = asyncio.run(run())
    assert max_concurrent_same_key == 1  # same-key sections never overlapped
    assert set(order) == {"a", "b", "c"}


def test_instances_do_not_share_locks():
    assert KeyedBuildLock()("k") is not KeyedBuildLock()("k")
