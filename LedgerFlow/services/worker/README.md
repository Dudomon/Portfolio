# LedgerFlow extraction worker

Go service that consumes `document.uploaded` events, extracts invoice fields from the uploaded
file and reports the result to the API. Message formats, queue names and the `ExtractionResult`
payload are defined in [`docs/CONTRACTS.md`](../../docs/CONTRACTS.md).

## Pipeline

```
RabbitMQ ──► queue.Consumer ──► pipeline.Processor
                                   1. decode + validate event (storage key must match tenant/document/sha256)
                                   2. skip if event_id is in the local LRU
                                   3. POST /internal/documents/{id}/status  {processing}
                                   4. GET object from S3/MinIO, verify sha256 while streaming
                                   5. extract.Router ──► nfe.Parser   (XML)
                                                     └─► llm.Client   (PDF, PNG, JPEG, non-NF-e XML)
                                   6. POST /internal/extractions      (ExtractionResult)
                                   7. remember event_id, ack
```

The API, not the worker, decides `extracted` vs `needs_review` from the submitted confidences and
the tenant threshold.

## Reliability semantics

- **At-least-once.** Messages are acked manually and only after their outcome is durable: the
  extraction was accepted by the API, the retry copy was confirmed by the broker (publisher
  confirms), or the DLQ copy was confirmed. A crash at any point causes redelivery, never loss.
- **Idempotency.** The API deduplicates `POST /internal/extractions` on `event_id` and is the
  source of truth. The in-memory LRU only avoids repeating a download and a paid model call when
  this replica sees a redelivery of something it already finished. A `409` when setting
  `processing` is tolerated for the same reason.
- **Retries.** A failed message is republished to `worker.document-uploaded.retry` with an
  `x-attempt` header and a per-message expiration (5s, 30s, 2m). On expiry the broker dead-letters
  it through the default exchange straight back to `worker.document-uploaded`, so other
  subscribers of `document.uploaded` do not see retries. The original is acked only after the
  retry copy is confirmed.
- **Dead letters.** After 3 retries (4 attempts), or immediately for permanent failures (invalid
  event, checksum mismatch, missing object, unsupported type, API `4xx` rejection), the message is
  copied to `worker.document-uploaded.dlq` with an `x-last-error` header and the document is
  marked `failed`. If that publish fails, the message is rejected and reaches the same DLQ through
  the main queue's `ledgerflow.dlx`.
- **Shutdown.** On SIGTERM the consumer is cancelled, prefetched-but-unstarted messages are
  nacked back to the queue, and in-flight messages get `SHUTDOWN_TIMEOUT` to finish. Anything
  still running after that is abandoned to redelivery.
- **Reconnects.** A lost connection or channel tears down the session and reconnects with jittered
  exponential backoff; `/readyz` reports not-ready meanwhile.

Known trade-off: RabbitMQ only expires messages at the head of a queue, so with one retry queue a
2-minute message can hold back a 5-second one behind it. The contract specifies a single retry
queue; one queue per delay tier would remove this if retry latency starts to matter.

## Why rules before the LLM for NF-e

An NF-e is a signed XML document with a published schema. Every contract field has a fixed XPath,
so a parser is exact, free and takes milliseconds; a model adds cost, latency and a chance of
misreading a value that is already machine-readable. The parser also checks what a model cannot be
trusted to: the access key's mod-11 check digit, the issuer CNPJ check digits, that the CNPJ and
invoice number embedded in the access key match the document, and that line items add up to
`ICMSTot/vProd`. A failed check sets that field's confidence to 0, which sends the document to
review instead of rejecting it.

The model handles what has no structure (PDF, scans) and XML that is not an NF-e. If the model
fails on content that turns out to be XML (a mislabelled upload), the router falls back to the
parser before giving up.

LLM output is constrained by a strict JSON schema and then validated: dates must be ISO dates,
amounts plain decimal strings (normalised with `shopspring/decimal`, never `float64`), CNPJs and
access keys must pass their check digits. Invalid values are kept for the reviewer but their
confidence is capped at 0.2, because the model's self-reported confidence is not calibrated.

## Cost tracking

Each response's `usage` is priced with a per-model table (USD per million tokens, overridable via
`LLM_PRICES`), matched by longest prefix so dated snapshots like `gpt-4o-mini-2024-07-18` resolve.
Cost is rounded up to whole micro-USD and:

- sent as `cost_usd_micros` on the extraction, with `model` and `prompt_version`;
- exported as `ledgerflow_worker_llm_cost_usd_micros_total{model}` and
  `ledgerflow_worker_llm_tokens_total{model,kind}`, including requests whose output was rejected,
  since those are billed too.

The prompt and its schema are embedded files versioned together as `invoice-v3`; a change to
either is a new version, so any extraction can be traced to the exact prompt that produced it.

## Observability

| Endpoint (`METRICS_ADDR`, default `:9090`) | Purpose |
|---|---|
| `/metrics` | Prometheus: messages processed/failed by outcome, extraction latency by extractor, LLM tokens and cost, Go runtime |
| `/healthz` | process is alive |
| `/readyz`  | consumer is attached to the broker |

Logs are JSON (`log/slog`). The event's `trace_parent` is extracted into the context, added to log
lines as `trace_id`/`span_id`, and propagated as `traceparent` on calls to the API.

## Running

Go 1.23+.

```sh
make test
cp .env.example .env && make run
make docker
```

Configuration is environment-only; see `.env.example` for every variable and its default.
