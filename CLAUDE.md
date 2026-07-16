# Repository guidance

Before making assumptions about Presidio or Harbor task internals, check the
current Harbor documentation at <https://www.harborframework.com/docs/>,
especially task structure, multi-step reward strategy, network policy, and
resource semantics.

Daytona sandbox contract (`src/presidio/environments/daytona.py`): task
`memory_mb`/`storage_mb` convert to Daytona whole-GiB units by rounding up
(never underprovision); every create path applies caller `sandbox_labels`
plus a `presidio.owner-token` ownership label used for orphan cleanup; 429s
are retried honoring `Retry-After`; auto-stop/auto-delete default to safe
nonzero behavior (60 min auto-stop, delete on stop).
