"""Process-wide, per-key build serialization.

When several trials of the same task start concurrently within one
``presidio run``, they all want to build the *same* named build artifact — a
Docker image, an E2B template, a Daytona snapshot. Building it more than once is
wasteful, and for backends that register a globally-unique *name* it is a hard
race: concurrent first-builds collide on the provider's uniqueness constraint
(e.g. E2B returns ``duplicate key value violates unique constraint`` when two
sandboxes try to register the same template alias at once).

:class:`KeyedBuildLock` serializes builds **per artifact name, within this
process**, so exactly one coroutine builds while the rest wait and then reuse
the now-cached artifact. Backends share one module-level instance per artifact
namespace instead of each re-rolling their own ``dict[str, asyncio.Lock]``.

It deliberately does **not** coordinate across processes — two separate
``presidio run`` invocations each have their own lock registry. A backend that
can still race across processes must *additionally* make its build idempotent
(treat the provider's "already exists" error as success and reuse the artifact).
The lock removes the common in-process race; idempotent-reuse covers the rest.
"""

from __future__ import annotations

import asyncio


class KeyedBuildLock:
    """A lazily-populated registry of per-key :class:`asyncio.Lock` objects.

    Acquire the lock for an artifact name around the build::

        BUILD_LOCKS = KeyedBuildLock()
        ...
        async with BUILD_LOCKS(name):
            await build(name)

    Locks are created on first use. ``dict.setdefault`` runs to completion
    without awaiting, so on a single event loop it is atomic — no guard lock is
    needed (this mirrors the original inline pattern in the Docker backend).
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def __call__(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())
