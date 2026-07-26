# ProofMode real-world benchmark

This harness evaluates the actual ProofMode service modules against downloaded,
hashed material from scikit-learn, MIT OpenCourseWare, the CDC archive, and an
optional NASA/JPL-Caltech infographic. It does not fine-tune Gemma and it does
not treat another model's opinion as ground truth. Gold topics, facts, rubrics,
and strong/misconception answers are inspectable in `cases.json`.

The default `quick` profile runs every subsystem once. The `full` profile adds
more files, the multimodal image, partial-credit answers, and three repeats.

```powershell
# Validate inputs, selected source tree, and live service without making calls
.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py --dry-run

# Download/cache/hash sources and exercise every subsystem
.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py --profile quick

# More expensive stability run
.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py --profile full
```

Downloads are cached under `benchmarks/.cache/`; timestamped raw outputs and
reports go under `benchmark_artifacts/`. Both paths are ignored by Git. Every
artifact includes source SHA-256 values, the selected code commit, model health,
case-level raw outputs, aggregate metrics, and baseline-to-optimized deltas.

## Controlled code comparison

Run the same harness and cache against a detached worktree, then compare the
hardened tree to its metrics. Product imports are delayed until after
`--repo-root` is applied.

```powershell
$cache = "$PWD\benchmarks\.cache"
$baseline = "C:\temp\proofmode-e33a732"

.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py `
  --repo-root $baseline --run-label baseline --cache-dir $cache

.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py `
  --repo-root $PWD --run-label optimized --cache-dir $cache `
  --compare-to .\benchmark_artifacts\<baseline-run>\metrics.json
```

Run the two trees in separate processes. Python module caching makes swapping
two versions of the same `proofmode` package inside one process unreliable.
Research search and page fetching remain live by default, allowing retrieval
latency changes such as bounded parallel fetches to be measured. Add
`--reuse-research-cache` to both runs when comparing only generation or
verification against an identical saved evidence pack.

## Metric interpretation

- Document extraction compares naive byte decoding with ProofMode's MIME-aware
  extractor using normalized gold-span recall.
- Curriculum mapping compares a heading heuristic with structured Gemma output
  using deterministic topic matching.
- Calendar scheduling compares naive sequential blocks with the deterministic
  conflict-aware scheduler.
- Assessment compares a keyword-coverage heuristic with Gemma rubric scoring
  against human-authored score labels.
- Research reports source recall, citation validity, unsupported-claim leakage,
  display coverage, repair rate, and latency for memory-only, raw-RAG, and the
  routed verification gate.
- Teach-back compares keyword/activity proxies with rubric scoring and genuine
  pre/post Teaching Impact.
- Gamification measures the score lift available from replay, one-topic farming,
  copied answers, low-transfer eligibility, and activity spam.

Gold-term recall is deliberately reported as a transparent proxy, not as full
semantic correctness. Educational-effect claims still require human evaluation
and a delayed learner study.
