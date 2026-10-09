# Evaluation design

The assistant answers policy questions that people act on: who approves a purchase, what
rate to withhold, whether an invoice can be paid. The failure that matters most is a
confident, well-cited answer that is wrong. The evaluation suite is built to catch that
failure and to tell which layer caused it.

## What is measured, and why each layer separately

| Layer | Metric | How it is computed | Question it answers |
| --- | --- | --- | --- |
| Retrieval | recall@k | Share of expected documents among the top *k* distinct retrieved documents | Did the right policy reach the model at all? |
| Retrieval | MRR | 1 / rank of the first expected document | Is the right policy near the top, where the reranker and context budget favor it? |
| Citations | citation precision | Share of cited documents that are expected sources | Does the answer point to the right policy? |
| Citations | citation recall | Share of expected documents that are cited | For multi-document questions, are all needed policies used? |
| Generation | groundedness | LLM judge, rubric `groundedness.v1`, scored 1-5 and normalized to 0-1 | Is every claim stated by the sources shown to the model? |
| Generation | answer relevance | LLM judge, rubric `answer_relevance.v1`, against a reference answer | Does the answer give the employee the decision, numbers and conditions they need? |
| Safety | abstention accuracy | Share of unanswerable questions where the assistant abstains | Does it say "not covered" instead of guessing? |
| Operations | mean cost, p95 latency | From the per-request cost ledger and wall-clock time | Is a quality gain worth its price? Reported, not gated. |

Retrieval metrics are deterministic and cheap. They are computed in two modes:

- `--mode retrieval` sends the raw question to the search backend with no planner and no
  query rewriting. A regression here points at ingestion, chunking or index
  configuration, not at prompts.
- `--mode full` runs the whole agent pipeline and scores the sources the answer agent
  actually received, so planning and query rewriting are included.

Comparing the two isolates the contribution of the planner and the rewriter.

Retrieval is scored at **document** level. The golden set names the documents that
contain the answer, not chunk ids, because chunk ids change whenever chunking is tuned
and the golden set has to survive those changes.

## Golden dataset

`evals/golden.jsonl` has 20 questions about the fictitious company's sample corpus:

- 15 single-document questions across procurement, accounts payable, tax and finance,
  phrased the way employees ask ("$30k", "W-8BEN", "PO").
- 3 questions that need two documents: a supplier bank-detail change, segregation of
  duties for supplier maintenance, and purchase approvals combined with withholding tax.
- 2 out-of-scope questions (parental leave, cryptocurrency) that must be refused.

Each item has the expected document ids and a short reference answer written from the
policy text. A unit test checks that every expected id exists in the corpus, so the
dataset cannot silently drift from the documents.

Twenty items are enough to exercise the pipeline and catch gross regressions; they are
not enough to estimate quality with tight confidence intervals. A production rollout
would grow the set from real (redacted) questions, labeled by policy owners, with
priority on questions that previously failed.

## LLM-as-judge

- **Versioned rubrics.** Rubrics live in `evals/rubrics/<name>.v<N>.md`. The highest
  version is loaded, and the report records each rubric's version and a content hash,
  so scores from different rubric revisions are never compared by accident. Changing a
  rubric means adding `v2`, not editing `v1`.
- **Anchored scales.** Each score from 1 to 5 has a concrete definition (for example,
  groundedness 3 means "one material claim is not supported"), which reduces drift
  between runs and between judge models.
- **Judge sees what the generator saw.** The groundedness judge receives the full text
  of the sources passed to the answer agent, not the whole corpus, so "grounded" means
  grounded in the evidence actually provided.
- **Separate judge deployment.** `JUDGE_DEPLOYMENT` is configurable so the judge can be a
  stronger model than the one under test, which limits self-preference bias.
- **Temperature 0 and strict JSON output.** Out-of-range or malformed judge output fails
  loudly (`JudgeError`) rather than being coerced into a score.
- **Calibration.** Before trusting a judge, score a sample of items by hand and check
  agreement. Re-run that check whenever the judge model or a rubric changes.

The runtime critic agent and the evaluation judge check similar properties on purpose,
but they are independent: the critic is a cheap gate in the request path, while the judge
is an offline measurement with a stronger model and a stricter rubric. If the critic
passes answers that the judge marks as ungrounded, the critic prompt needs work.

## Quality gates

`evals/thresholds.toml` defines minimum values per metric. `rag-eval` writes a Markdown
report and exits with code 1 when any applicable metric is below its threshold or was
not measured, so a CI job can block a merge or a deployment.

The thresholds are design targets, not measured baselines. Once a baseline exists against
a real index, each gate should be set a small margin below it: judge scores vary a
little from run to run even at temperature 0, and a gate set exactly at the baseline
produces flaky failures.

In CI, only the offline retrieval baseline runs on every push, because the full suite
needs Azure resources and costs money. The full suite is intended for a separate job
that authenticates to Azure with OIDC federation and runs on pull requests that touch
prompts, chunking or retrieval settings, and before each deployment.

## Offline baseline

`rag-eval --mode retrieval --offline` indexes the sample corpus into an in-memory BM25
index (keyword only: no vectors and no semantic ranker) and scores retrieval with no
Azure calls. On the 9-document sample corpus it reaches recall@5 of 1.000 and MRR of
0.944. Those numbers say little about production quality: with only nine documents,
almost any retriever finds the right one in the top five. They are useful as a smoke test
of the dataset and the metrics code, and as the floor that hybrid retrieval should beat
on a realistic corpus with near-duplicate and superseded policies.

## Known gaps

- No per-claim human labels yet, so judge accuracy is not quantified.
- No adversarial set for prompt injection in documents beyond the unit tests.
- Freshness (superseded policy versions) is not measured; a realistic corpus needs
  conflicting versions with effective dates and questions that test which one wins.
