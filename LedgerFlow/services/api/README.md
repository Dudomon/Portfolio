# LedgerFlow API

The HTTP API for LedgerFlow. It handles document intake, the review workflow, the audit
trail and the transactional outbox that hands work to `services/worker`. It also owns the
PostgreSQL schema. The interface it implements is specified in
[`docs/CONTRACTS.md`](../../docs/CONTRACTS.md).

## Layout

```
src/ledgerflow_api/
  main.py              app factory + lifespan (engine, S3 client, token service)
  config.py            pydantic-settings; validated at startup
  domain/              framework-free rules
    lifecycle.py         document state machine (explicit transition table)
    fields.py            extracted-field normalization + cross-field consistency rules
    tax_ids.py           CNPJ and NF-e access key check digits
    money.py             decimal parsing/formatting (no floats, no silent rounding)
  db/                  SQLAlchemy models, async engine, tenant-scoped sessions
  api/deps.py          auth, role guards, tenant session, service-token check
  api/v1/routers/      auth, me, documents, stats, internal
  services/            use cases: documents, ingestion (worker callbacks), audit,
                       outbox writer, storage, upload intake, stats
  workers/outbox_relay.py   outbox -> RabbitMQ publisher
  errors.py            RFC 9457 problem+json
  middleware.py        request id, access log, HTTP metrics
migrations/            Alembic; one initial revision with tables, enums, grants and RLS
tests/unit             pure tests (no I/O), including the error format through the real app
tests/api              HTTP tests against real PostgreSQL
```

## Key decisions

**Transactional outbox.** A request never talks to RabbitMQ. Uploads and admin retries insert
an `outbox` row in the same transaction as the document change. `ledgerflow-relay` claims
batches with `FOR UPDATE SKIP LOCKED` (so replicas can run side by side), publishes with
publisher confirms and `mandatory` routing, and marks rows published only after the broker
acks. Delivery is at-least-once. The event id is the outbox row id and is also sent as the
AMQP `message_id`, so consumers can deduplicate. If nothing is bound to the routing key yet,
the publish fails and the row stays pending; events are not dropped.

**Row-Level Security.** Every tenant-owned table has a `tenant_isolation` policy on
`app.tenant_id`. Request sessions run `SET LOCAL ROLE ledgerflow_app` and
`set_config('app.tenant_id', …, true)` from an `after_begin` hook, so every transaction is
scoped, including the ones that start after a commit. `ledgerflow_app` is not the table owner,
so the policies apply to it. Its grants also make `extractions`, `audit_events` and `outbox`
append-only. A handful of operations are cross-tenant by nature: login by email, the relay,
and the worker callbacks. These run as the connecting (owner) role and filter by id
explicitly. The tenant always comes from the JWT and never from the request body.

**Optimistic locking.** `documents.version` is SQLAlchemy's `version_id_col`, so every UPDATE
is a compare-and-swap. Reads return `ETag: W/"<version>"`. Corrections, approvals and
rejections require `If-Match`: a missing header returns `428` and a stale one returns `412`.
These handlers also lock the row (`SELECT … FOR UPDATE`) before they check the version, so
two writers with the same ETag can't both succeed.

**Lifecycle and review routing.** Every status change goes through `domain/lifecycle.py` and
writes an audit event with a `{status: {before, after}}` diff in the same transaction.
`overall_confidence` is the minimum field confidence. An extraction lands in `extracted` only
when that minimum is at or above the tenant's `auto_approve_threshold` (default 0.85) and the
rule checks pass: CNPJ check digits, NF-e key mod-11, issuer CNPJ inside the key, tax ≤ total,
line totals, due ≥ issue. Anything else goes to `needs_review`. Worker output is stored as
received, so reviewers see what the model produced. Human corrections are validated strictly
and stored as a new extraction version with `source: "human"` and confidence 1.0.

**Uploads.** The API reads the file in chunks with a hard size cap. It derives the content type
from magic bytes and ignores the client's `Content-Type`. Duplicates are found by
`(tenant_id, sha256)`, and the unique constraint also covers concurrent uploads of the same
file. The object is written to S3 before the row is inserted, so a crash can leave an
unreferenced object but never a row that points at a missing file. A reverse proxy should
still cap the request body, because multipart parsing happens before the handler runs.

**Idempotent callbacks.** `POST /internal/extractions` claims `event_id` in `processed_events`
with `INSERT … ON CONFLICT DO NOTHING` in the same transaction as its side effects, so a
redelivery is a no-op that returns `204`. Status callbacks that repeat the current status are
also no-ops.

## Running

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), PostgreSQL 16, RabbitMQ and an
S3-compatible store.

```bash
cp .env.example .env            # adjust URLs and secrets
uv sync
uv run alembic upgrade head
uv run ledgerflow-seed          # demo tenant: admin@ / reviewer@ / viewer@demo.example.com
uv run uvicorn ledgerflow_api.main:create_app --factory --reload
uv run ledgerflow-relay         # separate process
```

The seeded users share the password from `SEED_PASSWORD` (default `ledgerflow-demo`).
OpenAPI docs are at `/docs` outside production. Other endpoints: `/healthz` (liveness),
`/readyz` (database reachable) and `/metrics` (Prometheus). Tracing is enabled when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set.

The Docker image runs the API by default. Run `ledgerflow-relay` or `alembic upgrade head`
from the same image for the relay and for migrations.

## Testing

```bash
uv run pytest tests/unit        # no external services
uv run pytest                   # API tests start postgres:16 via testcontainers (needs Docker)
TEST_DATABASE_URL=postgresql+asyncpg://… uv run pytest   # or use a dedicated database
uv run ruff check . && uv run mypy -p ledgerflow_api
```

The API tests apply the real Alembic migration, so RLS policies and grants are covered as
well. Tables are truncated between tests. Never point `TEST_DATABASE_URL` at a database
whose data you need.
