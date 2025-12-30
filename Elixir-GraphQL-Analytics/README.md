# GraphQL Analytics Dashboard

A real time analytics platform built with Elixir, Absinthe GraphQL, and React. Demonstrates GraphQL subscriptions for live data updates and Dataloader for efficient batched queries.

## Why GraphQL for Analytics

### The REST Problem

A dashboard displaying user metrics, revenue data, and activity graphs requires multiple REST endpoints:
- GET /api/users/count
- GET /api/revenue/daily
- GET /api/activity/hourly
- GET /api/top-products

Four round trips. On a 200ms latency connection, that is 800ms before dashboard renders. Mobile users on cellular networks experience noticeable delay.

### GraphQL Solution

One request fetches all dashboard data:

```graphql
query Dashboard {
  userMetrics { totalUsers, newToday, activeNow }
  revenue { daily, monthly, yearToDate }
  activity { hourly { hour, count } }
  topProducts(limit: 5) { name, sales }
}
```

Single round trip: 200ms. The server processes queries in parallel using Elixir processes, so latency is determined by the slowest resolver, not the sum of all.

### Subscriptions for Real Time

Polling for live updates wastes bandwidth and battery. Every 5 seconds, client requests data that usually has not changed.

GraphQL subscriptions push updates only when data changes:

```graphql
subscription {
  metricsUpdated { activeUsers, ordersPerMinute }
}
```

Backend broadcasts metric updates to subscribed clients. Client receives update, re renders affected components. No polling, no wasted requests.

## Architecture

### Backend

Elixir processes continuously aggregate metrics from event streams. When a metric changes beyond a threshold, it broadcasts to subscribers.

```
EventStream -> MetricsAggregator (GenServer) -> PubSub -> Subscriptions -> Clients
```

Aggregation happens in memory for speed. Periodic snapshots persist to PostgreSQL for historical queries.

### Batching with Dataloader

N+1 queries kill analytics performance. A dashboard loading 50 products with their categories naively executes 51 queries (1 for products, 50 for categories).

Dataloader batches these into 2 queries regardless of result count. The resolver declares a data dependency; Dataloader collects all dependencies across the query tree and executes optimized batch queries.

### Query Complexity Analysis

Analytics queries can be expensive. A malicious query requesting nested data can overwhelm the server:

```graphql
query {
  products {
    orders {
      customer {
        orders {
          products { ... }
        }
      }
    }
  }
}
```

Absinthe's complexity analysis assigns cost to each field. Queries exceeding the budget are rejected before execution. This protects server resources without rate limiting legitimate queries.

## Features

1. **Dashboard Metrics**: User counts, revenue, activity graphs
2. **Real Time Updates**: Live metrics via subscriptions
3. **Time Series Data**: Hourly, daily, monthly aggregations
4. **Filtering and Grouping**: Flexible query parameters
5. **Query Cost Protection**: Complexity limits prevent abuse

## Project Structure

```
lib/
  analytics/
    metrics/
      aggregator.ex       # GenServer aggregating event stream
      snapshot.ex         # Periodic persistence to database
      calculator.ex       # Pure functions for metric calculations
    events.ex             # Event ingestion context
  analytics_web/
    schema.ex             # Root GraphQL schema
    schema/
      types.ex            # GraphQL type definitions
      subscriptions.ex    # Subscription definitions
    resolvers/
      metrics_resolver.ex # Query and subscription resolvers

frontend/
  src/
    components/
      MetricsCard.tsx     # Individual metric display
      TimeSeriesChart.tsx # Chart component with D3
      Dashboard.tsx       # Main dashboard layout
    hooks/
      useSubscription.ts  # GraphQL subscription hook
```

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

GraphQL endpoint: `http://localhost:4000/api/graphql`
GraphiQL explorer: `http://localhost:4000/graphiql`
