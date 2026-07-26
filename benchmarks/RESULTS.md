# ProofMode real-world benchmark results

**Test date:** 26 July 2026

**System under test:** local Gemma 4 E4B instruction model, Q4_0 GGUF, served by llama.cpp on Windows `10.0.26200` (`b10107-c0bc8591e`, four slots)

**Status:** real-source end-to-end benchmark plus deterministic adversarial tests; not a learner-outcomes study

## Executive result

ProofMode was exercised against real course material in machine learning, linear algebra, epidemiology, and solar-system scale. The benchmark covered file extraction, curriculum mapping, calendar planning and `.ics` round-tripping, question generation, answer assessment, live web research, claim verification, teach-back, and anti-reward-hacking scoring.

The clearest measured gains were:

- MIME-aware extraction increased mean gold-span recall from **0.6250 to 0.8125** (+18.75 percentage points) over four quick-profile files, and from **0.6429 to 0.8929** (+25 points) over seven scored full-profile prose/PDF sources.
- Conflict-aware calendar planning reduced overlap with existing events from **170 minutes to 0**, while retaining **100% exam detection, allocation, deadline compliance, and exported-block round-trip recall** in the realistic calendar fixture.
- A dedicated end-to-end rerun completed **16/16 records** and passed each real text/PDF/image input through a generated course map into a grounded quiz. All four optimized quizzes consumed their same-run upstream map and achieved **1.00 schema completeness, valid-MCQ rate, and unique-option rate**.
- Gemma rubric scoring reduced mean absolute error against six human-authored strong/misconception labels from **0.1667 to 0.0583** (65% lower); both methods made the same six pass/fail decisions.
- The hardened research path reduced displayed unsupported-claim leakage from **1.00 to 0.00** in the three-case benchmark. This is a safety improvement, not proof that every generated answer became more correct: strict verification also held answers it could not completely verify.
- After score-scale normalization, the live teach-back scorer's mean absolute error fell from the keyword baseline's **0.1833 to 0.0367** on the labelled pre/explanation/post case, while a no-learning activity attack fell from **100 reward points to 0**.
- ProofScore made replay and activity-spam attacks worth **0 additional score**, versus **+64** under naive XP, and held copied evidence. The original benchmark exposed one remaining flaw—an attempted transfer below the pass threshold could qualify—and a boundary regression now proves **0.59 is ineligible while 0.60 qualifies**.

These are descriptive results from small benchmark sets. They demonstrate working engineering and fail-closed behavior; they do **not** establish improved human retention, motivation, grades, or procrastination.

## Runs used

| Run | Scope | Result | Elapsed | Why it is included |
|---|---|---:|---:|---|
| `20260726T135756Z-baseline-clean` | quick, all 8 suites, detached clean commit `e33a7326...` | 55/56 records | 239.99 s | Controlled pre-hardening reference and all-system end-to-end run |
| `20260726T141106Z-optimized-reliability` | quick, course-map + research | 21/21 | 225.51 s | Planner normalization/retry and strict verifier behavior |
| `20260726T142103Z-optimized-final-affected` | quick, research + teach-back | 20/20 | 179.08 s | Verifier-capacity and score-normalization rerun |
| `20260726T142658Z-optimized-full-files` | full inputs; files, course-map, questions, assessment; 1 repeat | 55/56 | 318.32 s | Adds MIT problems/solutions, NASA page/image, partial and verbose misconception answers |
| `20260726T143355Z-optimized-assessment-retry` | full-input assessment rerun; 1 repeat | 24/24 | 134.11 s | Post-guard completion and scoring measurement; artifacts do not record whether the retry branch fired |
| `20260726T144552Z-optimized-e2e-course-quiz` | full-input course-map -> questions; 1 repeat | 16/16 | 135.15 s | Proves real files/image -> generated map -> grounded quiz chaining |

The optimized runs point at commit `e33a7326...` but explicitly record a **dirty working tree** in `environment.json`; the changes being measured were not yet committed. Raw rows, environment metadata, aggregates, and generated reports are retained under `benchmark_artifacts/<run-id>/` locally.

## Data and provenance

The harness downloads and hashes every selected input. Gold spans, topic aliases, expected answer concepts, strong answers, misconception answers, partial answers, and teach-back labels are inspectable in `benchmarks/cases.json`; Gemma does not write its own ground truth.

| ID | Real source | Bytes | SHA-256 (prefix) | Licence/use note |
|---|---|---:|---|---|
| `sklearn_common_pitfalls` | [scikit-learn common pitfalls](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/common_pitfalls.rst) | 25,066 | `6e2b42374fbb521c` | BSD-3-Clause ([licence](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/COPYING)) |
| `mit_eigen_summary` | [MIT OCW eigenvalue summary](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/1999c9f4accdbef05571a1014438f8dd_MIT18_06SCF11_Ses2.8sum.pdf) | 119,017 | `b339359c086e3714` | CC BY-NC-SA 4.0 ([terms](https://ocw.mit.edu/terms/)) |
| `mit_eigen_problems` | [MIT OCW problem set](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/45bdf363c6f1a564a61fdf61aa86748c_MIT18_06SCF11_Ses2.8prob.pdf) | 86,704 | `a52f04ffd7e107b7` | CC BY-NC-SA 4.0 |
| `mit_eigen_solutions` | [MIT OCW solutions](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/a5b090d92c330788febc5b2092906af9_MIT18_06SCF11_Ses2.8sol.pdf) | 106,624 | `95917825d06f32f1` | CC BY-NC-SA 4.0 |
| `cdc_epidemiology_lesson` | [CDC Principles of Epidemiology, Lesson 1](https://archive.cdc.gov/www_cdc_gov/csels/dsepd/ss1978/lesson1/index.html) | 80,515 | `f6e16c9d465d55af` | US government material; attribution retained ([policy](https://www.cdc.gov/other/agencymaterials.html)) |
| `cdc_epidemiology_pdf` | [CDC self-study course PDF](https://archive.cdc.gov/www_cdc_gov/csels/dsepd/ss1978/SS1978.pdf) | 6,305,461 | `30256d4fa0d56481` | US government material; attribution retained |
| `nasa_voyager_page` | [NASA PIA22921 description](https://science.nasa.gov/photojournal/voyager-2-and-the-scale-of-the-solar-system-artists-concept/) | 261,109 | `87037a14543af5df` | NASA media guidance; credit NASA/JPL-Caltech ([guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/)) |
| `nasa_voyager_scale` | [NASA/JPL-Caltech infographic](https://assets.science.nasa.gov/dynamicimage/assets/science/psd/solar/2023/09/p/i/a/2/PIA22921.jpg?crop=faces%2Cfocalpoint&fit=clip&h=1080&w=1920) | 325,865 | `074799652f606827` | NASA media guidance; visual labels only are scored |

## Methodology and metric boundaries

Two different comparisons are reported and should not be conflated:

1. **Within-run ablations** compare a deliberately simple baseline with the product path on the same case: byte decoding versus MIME extraction, heading matching versus Gemma mapping, sequential versus conflict-aware scheduling, keyword scoring versus rubric scoring, memory-only answer versus researched/verified answer, naive activity points versus ProofScore.
2. **Detached-tree comparisons** compare the clean `e33a7326...` tree with the optimized working tree. They share the local E4B server and downloaded source cache, but model sampling and live search remain variable. They are useful engineering measurements, not randomized statistical estimates.

The final aggregator reports completion over **every attempted row**, so failures cannot disappear when successful-row metrics are summarized. Within-run metric deltas use only identical successful `(case_id, repeat)` pairs and report `paired_n`; detached-run deltas require the same suite/variant/metric and equal sample count. Improvement signs come from explicit higher-is-better and lower-is-better allowlists (for example, fewer model calls is better), while unknown or unmatched metrics are omitted rather than assigned an inferred direction.

The server metadata recorded default generation at temperature **1.0**, `top_p=0.95`, and `top_k=64`; strict verification overrides temperature to **0.1**. The model advertised text, image, video, and audio modalities, although this benchmark exercised text plus one image only.

To fit the laptop server's 8K context, course-map requests now apply collection-wide caps rather than per-file truncation: **9,000 text characters, 1,500 exam-context characters, 24 evenly sampled text documents, and 2 images**. A single retry uses **4,500 / 750 characters and 1 image**. Fair water-fill allocation plus head-and-tail excerpts prevents the first large file from consuming the full text budget.

Metric meanings:

- **Gold-span/topic/term recall** is deterministic lexical coverage of inspectable expected concepts. It is a useful proxy, not semantic correctness.
- **Absolute error** compares a generated score with a human-authored benchmark label. The labelled set is small and was not independently double-rated.
- **Citation valid** means all extracted factual claims have syntactically known inline source IDs; it does not itself establish entailment.
- **Unsupported leakage** means an answer was displayed despite unsupported/uncertain claims or invalid citations. Zero is a display-safety result, not a factual-accuracy percentage.
- **Correct-or-held** counts a displayed answer only when gold-term recall is at least 0.60, while a safely held answer is counted as a safe outcome.
- **Teaching Impact and ProofScore** are product scoring functions, not validated psychometric scales.

The retained live artifacts predate the final failed-attempt telemetry patch. Their `model_call_count` and summed model latency can therefore undercount a malformed structured attempt that was retried; end-to-end `wall_ms` still includes that time. The released client now audits both the failed attempt and successful retry, with a regression proving two calls are recorded.

## Results by subsystem

### Files and curriculum mapping

Across four quick-profile inputs, naive byte decoding and MIME-aware extraction both returned nonempty output. Mean gold-span recall rose from **0.6250 to 0.8125**; the optimized median was **1.00**. Mean optimized extraction time was **1,915 ms**, dominated by the 6.3 MB CDC PDF (maximum **7,628 ms**); median time was only **15.8 ms**. The byte baseline was faster but often decoded container/binary data rather than useful document text.

The completed full-files run expanded this to all eight downloaded inputs. The image has no text-extraction gold and correctly produced no document text, so span recall is scored over the other seven sources: **0.6429 byte baseline versus 0.8929 MIME-aware**. All three MIT PDFs achieved **1.00 optimized span recall** (the problem PDF was **0.00** under byte decoding), and the NASA prose page achieved **1.00**. Mean optimized wall time fell to **987.6 ms** across all eight because seven were fast; the CDC PDF still took **7,841 ms**.

The clean run's heading heuristic achieved topic precision/recall/F1 of **0.2222 / 0.6667 / 0.3333**. Structured Gemma mapping reached **0.5833 / 0.6250 / 0.5857**, but only two of three model outputs completed; the CDC map was truncated. After adding a concise one-retry contract and deterministic numeric normalization, the reliability run completed all three maps with **100% schema completeness, valid fractional fields, bounded numeric fields, and valid estimated minutes**. Topic precision/recall/F1 in that stochastic rerun were **0.4429 / 0.5833 / 0.4966**, at a mean **26.80 s** per map. Reliability improved; topic matching did not improve monotonically.

With the NASA infographic added, all four full-profile maps completed and remained schema/numeric valid. Mean topic precision/recall/F1 were **0.5000 / 0.6625 / 0.5667**, versus the heading baseline's **0.1667 / 0.5000 / 0.2500**. On the image alone, the heading baseline found nothing; multimodal Gemma produced five topics and matched two of five visual gold concepts (**precision 0.40, recall 0.40, F1 0.40**): the heliopause/heliosphere concept and logarithmic distance scaling. That is evidence the real image path works, not a claim of comprehensive visual understanding.

### Calendar and reminders

The realistic `.ics` case contained four parsed events and a known assessment. Both schedulers found the assessment and allocated the requested study time. The sequential baseline overlapped existing events by **170 minutes**; ProofMode produced **0 conflict minutes**, **0 deadline violations**, and **0 block-limit violations**. Export and reparse recovered every planned block (**1.00 round-trip recall**) and preserved **12 alarms**. This measures scheduling mechanics, not whether a learner follows the reminders.

### Questions and assessment

For three quick-profile topics, generated question objects achieved **1.00 schema completeness, valid-MCQ rate, unique-option rate, and topic-term recall**. The optimized set had lower measured skill-label diversity (**1 versus 2**) and fewer rubric items (**3 versus 4**) than the generic baseline, so the test does not support a blanket claim of better question diversity. Mean generation time was **13.26 s**. The four-case full-files run retained all four 1.00 validity/coverage rates, increased mean rubric items to **3.75**, and averaged **10.08 s**, but skill diversity remained 1.

The later `optimized-e2e-course-quiz` run explicitly removed a benchmark shortcut: each optimized question case consumed the course map Gemma had generated from that case's real source earlier in the same run. This chained scikit-learn RST, MIT PDF, CDC HTML/PDF material, and the NASA infographic through **file/image -> course map -> quiz**. The run completed **16/16 total records**; all **4/4 optimized quizzes** recorded `upstream_course_map_used=true`, and schema completeness, valid-MCQ rate, and unique-option rate were each **1.00**. Optimized topic-name term recall was **0.50** versus the generic fallback's **1.00**, with mean optimized quiz latency **6.79 s**, **2.75** rubric items versus 4, and skill diversity 1 versus 2. The NASA quiz was structurally grounded in its upstream "Solar System Structure and Scale" map and asked about relative scale and the heliosphere, but its expected-name recall was **0.00** because it did not repeat the benchmark's named terms. This exposes both a lexical-metric limitation and incomplete named-concept coverage; it is not evidence of full factual correctness.

On six labelled answers (strong and misconception answers across all three subjects), keyword scoring had mean absolute error **0.1667** and Gemma rubric scoring **0.0583**. Both made all six pass/fail decisions correctly. Gemma added a mean **7.56 s** latency. Subsequent hardening accepts only finite numeric scores before normalizing valid decimal, 1–5, and percentage ranges; booleans, strings, NaN, infinities, and out-of-range values fail validation. Question payloads must contain 2–3 MCQs, exactly four string options per MCQ, an integer (not boolean) answer index from 0 through 3, a string open question, and 2–5 string rubric items. Malformed shapes are rejected rather than sliced into plausible-looking objects; assessment list fields alone are explicitly bounded, and malformed/truncated structured assessment output may be retried once.

The harder full-files assessment added partial and deliberately verbose misconception responses (12 labels total). Keyword MAE was **0.1817** with **91.67%** pass-decision accuracy. Gemma initially completed 11/12, with MAE **0.0845** on completed rows but only **81.82%** pass accuracy; one diagonalization misconception response ended in truncated JSON. A bounded retry guard was then added. The dedicated rerun completed **12/12**, with MAE **0.0900** (50.5% below the keyword error), **83.33%** pass accuracy, and mean wall time **11.18 s**. The artifacts record successful post-change completion but not whether any case invoked the retry, so they do not demonstrate a live retry activation. Gemma still made more threshold errors than the keyword baseline, so lower continuous-score error must not be presented as better classification.

### Research, search, and hallucination control

The original memory-only baseline displayed all three answers without valid citations and recorded unsupported leakage **1.00**. The old research verifier also had a false-safe failure mode: in the clean run it marked all three researched answers safe, even though two final reports show `used_llm_verifier=false`. Lexical overlap had been allowed to stand in for semantic entailment.

Hardening changed that behavior deliberately:

- deterministic fallback is always **uncertain**, never supported;
- every claim must have exactly one verifier row;
- a supported row requires a nonempty verbatim quote from an inline-cited source;
- unknown sources, missing/duplicate rows, negation, numbers, dates, units, directional conflicts, and a reason that disclaims support all downgrade or fail closed;
- verifier context contains only sources actually cited by extracted claims, capped at 3,500 characters each; verifier output allowance is 1,800 tokens at temperature 0.1;
- verifier reasons and quotes are validated as complete fields, never truncated before acceptance. Reasons over 600 characters or quotes over 2,000 invalidate the complete verifier pass, so a long suffix cannot hide a disclaimer or negation outside the checked prefix;
- after one repair, a deterministic selective-display step may copy only already-supported claim sentences. It creates no new prose and revalidates citations and quotes; if the verifier itself failed, nothing is selected.

In `optimized-reliability`, all three gold-term recalls were **1.00**, all citations were syntactically valid, and unsupported leakage was **0.00**. The strict gate displayed **0/3**: leakage had 3 supported + 7 uncertain claims, epidemiology 10 supported + 1 uncertain, and the diagonalization verifier output failed, leaving all seven claims uncertain. This is the intended safe failure, but poor usability. Increasing verifier capacity and reducing context made the semantic verifier complete on two of three cases in `optimized-final-affected`; one narrowed answer displayed via selective excerpt and unsupported leakage remained **0.00**. In that same run, `research-diagonalization` exercised the selective path: **1/3** optimized research answers was rescued as a quote-backed excerpt, while **2/3** remained held and unsupported leakage stayed **0.00**. The displayed excerpt's gold-term recall was **0.50**, below the 0.60 approximate-correctness threshold. It therefore demonstrates a safe partial recovery, not full correctness. Final citations were still valid for only **1/3** cases and mean answer-path time was **33.81 s**.

Search routing in `optimized-reliability` moved target-domain hit rate from **0.3333 to 0.6667**, authority signal from **2.617 to 3.833**, and fetched-page fraction from **0.6667 to 0.7778**. Mean retrieval time fell from **3,271 ms to 2,287 ms** (30.1%) within that run. Because search results and network conditions changed, this is not a clean causal estimate. A deterministic six-source test directly verifies the optimization mechanism: page fetches overlap with a maximum of four active workers, two source failures remain isolated, and source/warning order stays stable.

### Teach-back and score normalization

The clean run exposed a scale error: Gemma emitted a teaching-quality value on a 1–5 scale where the code expected 0–1, producing mean absolute error **1.1367**. After normalization and bounded retry, `optimized-final-affected` estimated pre-score **0.25**, quality **0.95**, and post-score **1.00**, for mean absolute error **0.0367** against the human labels (**0.25 / 0.92 / 0.92**) and the correct gain direction. This is one scenario, so it demonstrates recovery of score semantics rather than general grading validity.

Naive activity reward gave both genuine teaching and a no-learning attack **100**. Learning-lift reward gave the genuine case **64.9** and the no-gain attack **0**. A lower genuine value is intentional calibration, not reduced learning.

### ProofScore and anti-reward-hacking

ProofScore weights are knowledge 32%, transfer depth 24%, calibration 10%, topic breadth 20%, teaching impact 7%, and reliability 7%; unavailable optional components are renormalized. Public eligibility requires at least four verified checks across two topics, two retention checks after at least 20 hours scoring at least 0.60, at least one transfer check scoring at least 0.60, and no integrity hold. Study time, message count, note length/style, and presumed AI authorship add no score.

In the clean quick benchmark, replaying evidence and adding activity spam each increased naive XP by **64**, but increased optimized ProofScore by **0**; **20 replay events were held**, copied evidence triggered a hold, and the un-attacked evidence remained leaderboard eligible. The run also found that a low transfer attempt could still satisfy the public gate (`low_transfer_false_eligibility=1`). The gate now uses the same 0.60 pass threshold as delayed evidence; the focused adversarial boundary test confirms **0.59 → provisional/ineligible** and **0.60 → verified/eligible** when all other evidence is identical.

## Adversarial and regression evidence

The final regression count is **199 passed tests**. Relevant adversarial coverage includes:

- overconfident-verifier conflicts covering disclaimed support, negation, transposed numbers, units, month/date mismatches, opposite directional terms, and long-tail attempts to hide a conflict beyond a bounded prefix;
- four incomplete/duplicate/unknown/invalid verifier-row modes, a fabricated non-verbatim quote, an unknown inline citation, and semantic-verifier failure—all held;
- selective-display tests proving mixed supported/uncertain output copies only supported text, while verifier failure or zero supported claims yields no excerpt;
- replayed retrieval fingerprints, one-topic farming, copied peer answers, reciprocal teaching caps, provisional high scores, failed delayed retrieval, and the 0.59/0.60 transfer boundary;
- score normalization across decimal, 1–5, and percentage values; strict finite-number, MCQ cardinality/type, option, rubric, and answer-index validation; list bounds; and one bounded structured-output retry guard;
- model-call telemetry proving a malformed structured attempt plus its successful retry records two calls and both latencies without storing malformed model prose;
- global planner caps, fair multi-file allocation, head/tail excerpts, evenly sampled long upload sets, bounded image counts, and smaller retry context;
- bounded concurrent fetching with preserved rank/source IDs and deterministic failure warnings.

These tests validate explicit invariants. They do not simulate collusion at scale, sophisticated paraphrase attacks, or long-term changes in learner behavior.

## What changed and what the measurements say

| Optimization | Evidence of improvement | Cost or unresolved issue |
|---|---|---|
| MIME-aware extraction | +18.75 pp mean span recall | Large PDFs dominate latency; scanned PDFs still need OCR |
| Planner numeric contract, normalization, one retry | 3/3 valid maps versus one truncation in clean run | Mean 26.80 s; stochastic topic F1 was 0.4966, below the earlier 0.5857 two-case result |
| Global planner request caps and fair sampling | Unit tests bound text/exam/image inputs; the post-change end-to-end run completed all 4/4 real course maps | Caps trade exhaustive document coverage for predictable 8K-context requests; no causal before/after latency pair |
| Conflict-aware scheduler and `.ics` alarms | 170 → 0 conflict minutes; perfect fixture round trip | One deterministic calendar scenario, not a live Google Calendar account test |
| Rubric scoring + strict payload validation | answer-score MAE 0.1667 → 0.0583; teach-back MAE 0.1833 → 0.0367; assessment rerun completed 12/12 | Small authored label set; retry activation was not logged; strict rejection may surface more held/error states |
| Routed sources + four-worker retrieval | stronger domain/authority proxies; 3.271 → 2.287 s in one live run | Live-web variance prevents causal latency attribution |
| Complete quote-backed verifier | unsupported display leakage 1.00 → 0.00 | More calls and holds; final affected run averaged 33.81 s |
| Bounded verifier context/output | semantic verifier completed 2/3 final-affected cases | Long multi-claim JSON still fails sometimes on E4B |
| Whole-field verifier bounds | Long suffixes cannot be silently truncated past contradiction checks; adversarial regressions fail closed | A reason over 600 characters or quote over 2,000 invalidates the entire verifier pass |
| Selective supported excerpt | Live diagonalization case displayed 1/3; 2/3 held; leakage 0.00 | Displayed excerpt had 0.50 gold-term recall, so recovery was partial rather than fully correct |
| ProofScore anti-farming + transfer pass gate | replay/spam +64 → 0; copied evidence held; 0.59 boundary rejected | Not a validated measure of actual knowledge or fairness across populations |

## Reproduction

From the repository root in PowerShell, with the local Gemma server healthy at `http://127.0.0.1:8080/v1`:

```powershell
# Verify selected sources and model health without generation
.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py --dry-run

# Reproduce the quick all-system run
.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py `
  --profile quick --suite all --run-label reproduction-quick

# Full files, image path, partial answers, and three repeats
.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py `
  --profile full --suite all --run-label reproduction-full

# Deterministic regression/adversarial suite
.\.venv\Scripts\python.exe -m pytest -q
```

For a detached-tree comparison, run each tree in a separate Python process:

```powershell
$cache = "$PWD\benchmarks\.cache"
$baseline = "C:\path\to\clean-baseline-worktree"

.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py `
  --repo-root $baseline --run-label baseline-reproduction --cache-dir $cache

.\.venv\Scripts\python.exe .\benchmarks\run_benchmark.py `
  --repo-root $PWD --run-label optimized-reproduction --cache-dir $cache `
  --compare-to .\benchmark_artifacts\<baseline-run>\metrics.json
```

Use `--reuse-research-cache` on both runs when the question is model/verification behavior with fixed evidence. Leave it off when measuring the fetch implementation, and retain the raw packs because live search is not deterministic.

## Limitations and next measurements

- All cited runs have **one repeat** and tiny case counts; no confidence intervals or significance claims are justified. The harness defaults to three repeats for `--profile full`, but the retained full-files run explicitly overrode this to one.
- E4B is both writer and verifier, so errors are correlated even though the passes use different prompts and the verifier is deterministic where possible.
- Term recall can reward keyword presence without correct relationships and can miss valid paraphrases. The NASA upstream-grounded quiz scored 0.00 expected-name recall despite testing map-level scale/heliosphere relationships; that also leaves named-concept coverage incomplete, so neither lexical recall nor structural grounding alone proves correctness.
- The assessment labels were authored for this benchmark, not independently rated by multiple educators.
- Web ranking uses heuristic authority signals. An official domain can still contain irrelevant material, and live results can disappear or change.
- PDF extraction currently truncates document context and does not OCR scanned pages. Planner input is globally capped at 9,000 text and 1,500 exam-context characters (smaller on retry), with at most 24 text documents and 2 images. Fair sampling limits early-file bias but may omit relevant middle details. The NASA image produced no extraction text but did exercise the multimodal course-map path; visual results are reported separately from prose-source results.
- The calendar benchmark validates `.ics` parsing/export and conflict logic, not OAuth, live calendar writes, notification delivery, or student compliance.
- ProofScore robustness is measured against specified attacks, not proven strategy-proof. Accessibility, cultural fairness, and subject-to-subject calibration remain unmeasured.
- No students were followed over time. The next credible product study needs consented learners, delayed tests, intervention/no-intervention comparison, reminder burden, false-hold review, and subgroup analysis.

The defensible conclusion is therefore narrow: **ProofMode's prototype works end to end on real material, its main safety and reward-hacking invariants are exercised, and several engineering optimizations improved measured proxy outcomes. Real learning and behavior impact remain open empirical questions.**
