# Synthetic retrieval replay

`MemoryService.retrieve_query` accepts optional `exclude_task_ids` and `rng`
keyword arguments. Exclusion compares string-normalized IDs against the existing
`task_id` / `sample_index` / `id` metadata selector and `source_task_id`. Numeric
zero remains a valid ID. Filtering happens before Q statistics, greedy selection
and exploration. An excluded query bucket is replaced by the next eligible
bucket within the similarity threshold, preserving the meaning of `k` as the
number of query buckets. With no exclusions, existing retrieval behavior remains.

The `memrl.evaluation.run_retrieval_replay` helper consumes a `memoryeval` v1
envelope and a prebuilt service. It performs no LLM invocation and does not score
the fixture's trajectory, diagnosis label or reward. Its output concerns retrieval
isolation and reproducibility; it provides no evidence of clinical effectiveness
or improvements in model capability.

The envelope requires:

- `schema_version: 1`, `synthetic: true`;
- `source: {sha256, format, generator, version}` (plus optional generator seed);
- `partitions: {train: [patient_id, ...], evaluation: [patient_id, ...]}` with
  disjoint patients;
- `tasks`, each with `task_id`, `source_task_id`, `patient_id`, `split`, and `query`.
  Splits are `adapt`, `replay`, `heldout`, or `drift`. Train patients belong to
  `adapt`; all other splits use evaluation patients. Multiple encounters for a
  patient must stay in one split. Optional `trajectory`, `reward` and `metadata`
  can preserve targets for other consumers; this helper never scores them.

Populate the service only from `adapt` tasks. Keep `task_id`, `source_task_id`,
`patient_id`, and `split="adapt"` in each stored memory's metadata. Every candidate
in an evaluation retrieval must have this provenance. A candidate from the
evaluation patient's partition, or without adapt provenance, produces
`partition_leakage` and an empty action list. A returned source-task match produces
`source_task_leakage`. Exceptions produce `retrieval_error` with an exception type,
and an action missing from the candidate set produces `invalid_retrieval_result`.
Errors are never represented as successful empty predictions.
Candidates must also reference a known adaptation source task; missing or unknown
origin IDs produce `memory_provenance_error`. The replay calls retrieval with
`raise_on_load_error=True`, so failed or missing storage reads fail the record even
if another candidate loaded successfully. The ordinary retrieval default retains
its existing best-effort behavior.

```python
import json
from pathlib import Path
from memrl.evaluation import run_retrieval_replay

manifest = json.loads(Path("memoryeval.json").read_text())
# `service` is an initialized MemoryService populated only from adapt tasks.
report = run_retrieval_replay(
    service,
    manifest,
    storage={"backend": "MemoryOS-local", "version": "1.0.0",
             "integration": "describe the actual storage test here"},
    seed=42,
    k=5,
    # If available, snapshot actual provider counters AFTER retrieval:
    provider_usage=lambda: {"provider": "fixed-fixture-vectors", "model_calls": 0,
                            "embedding_calls": embedder.calls, "cost_usd": 0},
    source_bytes=Path("cases.json").read_bytes(),
)
Path("retrieval-replay.json").write_text(json.dumps(report, indent=2))
```

Caller-supplied storage provenance identifies the actual backend/version and
integration evidence; a fixture store must say so. The output includes the source
digest, canonical manifest and records SHA256 digests, seed, `k`, task/source IDs,
query hashes, exclusion lists, candidates, actions and failure categories. Passing
source bytes verifies the declared source digest; omitting them records
`source_sha256_verified=false`. Repeated retrieval against the same fixed store,
vectors and configuration uses a local RNG and does not change global RNG state.

Provider counters and cost are caller-reported. Unknown or inapplicable values
remain `null`; no token or price estimate is invented. A usage callback is sampled
after the run. Latency and wall-clock timestamps are omitted from deterministic
records. Run retrieval synchronously and keep the store unchanged during a replay.

Focused offline regressions:

```sh
python -m unittest discover -s tests -p test_task_exclusion.py -v
python -m unittest discover -s tests -p test_offline_replay.py -v
```

These tests execute the real retrieval/scoring code with fixed vectors and an
in-memory persistence fixture, replacing only external MemOS import types. They
do not establish a complete MOS-service or model-provider integration.
