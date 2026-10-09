# ADR 0002 — Deterministic parsing before LLM extraction

- Status: Accepted
- Date: 2026-10-09

## Context

Brazilian NF-e invoices arrive as signed XML with a fixed schema. Supplier PDFs and scans have no schema.
Sending everything to an LLM is simple but costs money per document, adds seconds of latency and
introduces non-determinism into data that is already structured.

## Decision

The worker routes by content type:

1. **NF-e XML → rule-based parser.** Fields come from the schema; confidence is `1.0`, `source = "rule"`.
   The 44-digit access key and CNPJ check digits are validated; a failed check lowers confidence so the
   document lands in review instead of being auto-approved.
2. **PDF / image → LLM** with a versioned prompt (`invoice-v3`) and a strict JSON schema response format.
   Returned values are re-validated (dates, decimal strings, CNPJ); invalid values have their confidence
   downgraded rather than being silently accepted.

`prompt_version` and `model` are stored on every extraction so results stay reproducible and comparable
when the prompt or model changes.

## Consequences

- NF-e documents cost nothing to extract and are fully deterministic.
- LLM output is never trusted blindly: validation is the same code path for both sources.
- Two extractors to maintain, behind one `Extractor` interface.
