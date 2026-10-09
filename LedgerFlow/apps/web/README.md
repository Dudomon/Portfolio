# LedgerFlow web

Reviewer-facing front end for LedgerFlow: upload supplier invoices, follow extraction status, and
correct / approve / reject low-confidence results. The API contract it consumes is
[`docs/CONTRACTS.md`](../../docs/CONTRACTS.md).

React 18 · TypeScript (strict) · Vite · TanStack Query · React Router 6 (data router) ·
React Hook Form + Zod · CSS Modules · Storybook 8 · Vitest + Testing Library + MSW.

## Running it

```sh
npm install
npm run dev:mocks   # no backend needed: MSW serves the API from in-memory demo data
npm run dev         # proxies /api to http://localhost:8000 (see .env.example)
```

With mocks, sign in as `reviewer@demo.test`, `admin@demo.test` or `viewer@demo.test` with any password.

| Script                          | Purpose                                     |
| ------------------------------- | ------------------------------------------- |
| `dev` / `dev:mocks`             | Vite dev server, against the API or MSW     |
| `build`                         | Type-check, then production build to `dist` |
| `typecheck`                     | `tsc` for app and tooling configs           |
| `lint` / `format`               | ESLint (type-aware, a11y) / Prettier        |
| `test` / `test:watch`           | Vitest                                      |
| `storybook` / `build-storybook` | Component workbench for `src/ui`            |

`docker build -t ledgerflow-web .` produces an unprivileged nginx image serving `dist` on port 8080
with SPA fallback and security headers. Set `CSP_CONNECT_SRC` to the object-storage origin so
document previews (presigned URLs) are allowed by the Content-Security-Policy.

## Structure

```
src/
  api/        HTTP client, Zod schemas mirroring the contract, query keys, TanStack Query hooks
  app/        composition root, providers, routes, layout, route error boundaries
  features/
    auth/       login, in-memory session store, RequireAuth / RequireRole
    documents/  list page, URL-synced filters, upload dialog
    review/     review workbench, field model, audit timeline, conflict handling
    dashboard/  stats tiles
  lib/        decimal arithmetic, formatting, small hooks
  mocks/      MSW handlers + deterministic fixtures (dev and tests)
  styles/     design tokens and global CSS
  ui/         in-house component library (each component has stories)
  test/       test setup and render helpers
```

Dependencies point one way: `ui` and `lib` know nothing about the domain, `api` knows the contract
but not React components, `features` compose both, and `app` wires everything together.

## Design decisions

**In-house design system.** `src/ui` holds the primitives (Button, TextField, Select, Badge,
ConfidenceMeter, DataTable, Dialog, Toast, Tabs, FileDropzone, Skeleton…). Styling is CSS Modules on
top of design tokens in `styles/tokens.css`: a raw palette plus a semantic layer
(`--color-surface`, `--color-text-muted`, …) that components use exclusively, so dark mode is a
remap of the semantic layer only. Components follow the WAI-ARIA patterns they implement: the
dialog traps focus, handles Esc and restores focus; tabs use roving tabindex with arrow keys; the
data table exposes `aria-sort` and offers row navigation with a single tab stop; the confidence
meter is a `role="meter"` with a spoken value.

**Validated API boundary.** Every response is parsed with Zod schemas that mirror the contract.
Errors become typed classes: `ApiError` carries the RFC 9457 problem document,
`PreconditionFailedError` (412) is distinct so the UI can offer a reload instead of a retry,
`NetworkError` and `ResponseValidationError` cover the remaining failure modes. Every request sends
an `X-Request-ID`; the audit timeline shows the request id stored with each event.

**Optimistic concurrency.** Document reads keep their `ETag`. Review mutations send `If-Match` with
the tag of the snapshot the reviewer's form was built from, not the latest cached copy, so a
background refetch can never turn a stale edit into a silent overwrite. Field corrections are
applied optimistically and rolled back on failure; a 412 opens a dialog that reloads the latest
version.

**Money as decimal strings.** Amounts never pass through `parseFloat`. `lib/decimal.ts` does the
little arithmetic the UI needs (line totals, sums, rounding) on scaled `BigInt`s, and
`Intl.NumberFormat` formats the decimal strings directly.

**State placement.** Server state lives in TanStack Query; list filters live in the URL; form
state lives in React Hook Form; the access token lives in memory only (no storage), at the cost of
signing in again after a reload. The contract has no refresh endpoint, so token expiry or any 401
ends the session and returns to the login page, preserving the target location.

**Review workbench.** Document preview and fields side by side (tabs below 1024px). Fields below
the 0.85 threshold are highlighted and show their confidence and source. Line items are editable
with per-line and total consistency hints. Shortcuts: `a` approve, `r` reject, `j` / `k` next /
previous document in the review queue. Leaving with unsaved corrections asks for confirmation.

## Testing

Tests run against the same MSW handlers used in development, so they exercise the real client,
cache and contract parsing. Covered: decimal arithmetic, API error mapping, DataTable
(sorting, keyboard navigation, states, pagination), ConfidenceMeter, and the review workbench
(correcting a field, validation, approving, the 412 conflict flow, read-only role).

Tests use happy-dom rather than jsdom: jsdom's `AbortSignal` is rejected by Node's `fetch`, which
breaks cancellable queries under MSW.

## Contract notes

- `GET /me` returns a `User` that the contract does not define. The client expects
  `{ id, name, email, role }` and falls back to the JWT `sub` claim if the call fails.
- The tenant-specific auto-approval threshold is not exposed by the API; the UI highlights against
  the contract default of 0.85.
