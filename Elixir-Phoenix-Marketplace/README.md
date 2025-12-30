# Circular Marketplace

A real time consignment marketplace built with Elixir, Phoenix, Absinthe GraphQL, and React. Designed for sustainable commerce where users buy and sell pre owned luxury items.

## Why This Stack

### Elixir and Phoenix

Elixir runs on the BEAM virtual machine, the same runtime that powers Ericsson telecom switches with 99.9999999% uptime. This matters for a marketplace because:

1. **Concurrent connections**: Each user session runs in an isolated lightweight process. A single server handles 2 million concurrent WebSocket connections. Traditional thread per connection models exhaust memory at 10,000 connections.

2. **Fault isolation**: If one user's session crashes (malformed input, network hiccup), other users are unaffected. The supervisor restarts the failed process in microseconds. In Ruby or Node, an unhandled exception can bring down the entire server.

3. **Real time by default**: Phoenix Channels use the same process model. Broadcasting price updates to 50,000 connected clients is a single function call, not a Redis pub/sub infrastructure project.

### GraphQL over REST

REST requires multiple round trips for related data. Fetching a product with seller info and reviews means three requests. GraphQL fetches exactly what the client needs in one request. For mobile users on slow connections, this reduces latency from 900ms to 200ms.

Subscriptions provide real time updates without polling. When a product sells, all users viewing that product receive an instant update through their existing WebSocket connection.

### React with TypeScript

GraphQL schema generates TypeScript types automatically. When the backend adds a field, the frontend compiler catches every place that needs updating. This eliminates an entire class of runtime errors.

## Architecture Decisions

### Context Boundaries

The codebase separates into three contexts: Accounts, Catalog, and Orders. Each context owns its data and exposes a public API. The Orders context never queries the users table directly; it calls `Accounts.get_user/1`.

This matters because:
1. Database schema changes in Accounts only require updates within that context
2. Each context can be extracted to a separate service if scale demands it
3. New developers understand boundaries immediately

### Soft Deletes for Audit Trail

Products use `deleted_at` timestamp instead of hard deletes. Luxury consignment requires provenance tracking. When a dispute arises six months later, we need the original listing data.

### Decimal for Money

All prices use the Decimal type, not floats. `0.1 + 0.2` equals `0.30000000000000004` in float arithmetic. For a $10,000 watch, float rounding errors become real money.

## Project Structure

```
lib/
  marketplace/           # Business logic, no HTTP concerns
    accounts/            # User registration, authentication
    catalog/             # Products, categories, search
    orders/              # Purchases, payments, shipping
  marketplace_web/       # HTTP layer, GraphQL, Channels
    schema/              # Absinthe GraphQL types
    resolvers/           # Query and mutation implementations
    channels/            # Real time WebSocket handlers

frontend/
  src/
    components/          # Reusable UI components
    graphql/             # Queries, mutations, generated types
    pages/               # Route components
    hooks/               # Custom React hooks
```

## Data Model

Users have many Products (as sellers) and many Orders (as buyers). Products belong to one Category. Orders contain line items referencing Products with quantity and price at time of purchase (prices change, order history must not).

## Running Locally

```bash
# Backend
mix deps.get
mix ecto.setup
mix phx.server

# Frontend
cd frontend
npm install
npm run dev
```

## Testing Strategy

Unit tests cover context functions in isolation with mocked dependencies. Integration tests verify GraphQL queries return expected shapes. Property based tests generate random products and verify search always returns relevant results.

No end to end browser tests. The cost of maintaining Selenium tests exceeds the bugs they catch. Instead, TypeScript catches UI data mismatches at compile time.
