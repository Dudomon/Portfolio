import {
  ApolloClient,
  InMemoryCache,
  createHttpLink,
  split,
  ApolloLink,
} from "@apollo/client";
import { setContext } from "@apollo/client/link/context";
import { GraphQLWsLink } from "@apollo/client/link/subscriptions";
import { getMainDefinition } from "@apollo/client/utilities";
import { createClient } from "graphql-ws";

/**
 * Apollo Client configuration with authentication and WebSocket subscriptions.
 *
 * Uses split link to route:
 * - Queries and mutations over HTTP
 * - Subscriptions over WebSocket
 *
 * This separation exists because HTTP is stateless and works well with
 * load balancers, while WebSocket requires persistent connections for
 * real time updates.
 */

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:4000/api";
const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:4000/socket";

// Storage key for auth token. Using constant prevents typos across codebase.
const TOKEN_KEY = "marketplace_token";

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearAuthToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

// HTTP link for queries and mutations
const httpLink = createHttpLink({
  uri: `${API_URL}/graphql`,
});

// Auth link injects token into every request
const authLink = setContext((_, { headers }) => {
  const token = getAuthToken();

  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : "",
    },
  };
});

// WebSocket link for subscriptions
// Lazy initialization: connection opens on first subscription, not on page load
const wsLink = new GraphQLWsLink(
  createClient({
    url: `${WS_URL}/graphql`,
    connectionParams: () => {
      const token = getAuthToken();
      return token ? { authorization: `Bearer ${token}` } : {};
    },
    // Reconnect on connection loss with exponential backoff
    retryAttempts: 5,
    shouldRetry: () => true,
  })
);

// Route subscriptions to WebSocket, everything else to HTTP
const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query);
    return (
      definition.kind === "OperationDefinition" &&
      definition.operation === "subscription"
    );
  },
  wsLink,
  authLink.concat(httpLink)
);

// Cache configuration with type policies
const cache = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        // Merge paginated product results instead of replacing
        products: {
          keyArgs: [
            "categoryId",
            "sellerId",
            "minPrice",
            "maxPrice",
            "condition",
            "search",
            "sort",
          ],
          merge(existing, incoming, { args }) {
            // First page replaces; subsequent pages append
            if (!args?.page || args.page === 1) {
              return incoming;
            }
            return {
              ...incoming,
              items: [...(existing?.items || []), ...incoming.items],
            };
          },
        },
      },
    },
    Product: {
      // Products identified by id for cache normalization
      keyFields: ["id"],
    },
    User: {
      keyFields: ["id"],
    },
    Order: {
      keyFields: ["id"],
    },
  },
});

export const apolloClient = new ApolloClient({
  link: splitLink,
  cache,
  defaultOptions: {
    watchQuery: {
      // Fetch from cache first, then network for fresh data
      fetchPolicy: "cache-and-network",
      // Return partial data while loading related fields
      returnPartialData: true,
    },
    query: {
      // Single queries use network with cache update
      fetchPolicy: "network-only",
    },
  },
});
