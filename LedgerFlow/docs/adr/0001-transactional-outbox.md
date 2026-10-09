# ADR 0001 — Transactional outbox for document events

- Status: Accepted
- Date: 2026-10-09

## Context

An upload must (a) persist the document row, (b) write an audit event and (c) notify the worker.
Publishing to RabbitMQ inside the request handler creates a dual-write: the DB commit can succeed while
the publish fails (document stuck in `uploaded` forever), or the publish can succeed for a transaction that
later rolls back (worker processes a document that does not exist).

## Decision

The API writes the event to an `outbox` table in the same transaction as the state change. A relay process
claims pending rows with `SELECT ... FOR UPDATE SKIP LOCKED`, publishes with publisher confirms, and marks
them sent. Delivery is therefore **at-least-once**; consumers deduplicate on `event_id`
(`processed_events` table in the API, LRU fast-path in the worker).

## Consequences

- No lost or phantom events; state and notification are atomic.
- Extra moving part (relay) and a few hundred ms of added latency from polling — acceptable for a
  workflow measured in seconds.
- `SKIP LOCKED` lets the relay scale horizontally without coordination.
- Outbox rows are pruned after 7 days by a scheduled job.

## Alternatives considered

- **CDC (Debezium) on the outbox table** — lower latency, but adds Kafka Connect to a stack that does not
  otherwise need it.
- **Publish-then-commit with compensations** — pushes failure handling into every call site.
