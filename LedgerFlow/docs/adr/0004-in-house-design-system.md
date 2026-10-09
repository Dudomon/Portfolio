# ADR 0004 — In-house component library instead of a UI kit

- Status: Accepted
- Date: 2026-10-09

## Context

The review workbench is the product: reviewers spend hours a day in it, so density, keyboard flow and
predictable focus behaviour matter more than visual novelty. Off-the-shelf kits are optimised for generic
CRUD screens and bring large style overrides once the design diverges.

## Decision

Build a small library in `apps/web/src/ui` — tokens as CSS custom properties (light/dark), CSS Modules,
and only the components the product needs (DataTable, Dialog, ConfidenceMeter, FileDropzone, Toast, …).
Every component ships a Storybook story and is tested with Testing Library using accessible queries.

## Consequences

- Full control over accessibility (focus trap, roving focus in tables, ARIA live regions for toasts).
- Small bundle; no theme-override layer.
- The team owns maintenance — the scope is kept deliberately narrow, and anything generic (date pickers,
  rich text) would be adopted from a headless library rather than built.
