/**
 * GraphQL Subscription Hook for Phoenix/Absinthe WebSocket
 *
 * Architecture Overview:
 *
 * This hook manages the complete lifecycle of GraphQL subscriptions over
 * Phoenix Channels. Unlike HTTP based GraphQL clients (Apollo, urql), Phoenix
 * uses its own WebSocket protocol with multiplexed channels rather than
 * the graphql-ws specification.
 *
 * Connection flow:
 * 1. Client connects to ws://host/socket/websocket
 * 2. Phoenix assigns a socket ID and establishes heartbeat
 * 3. Client joins the "__absinthe__:control" channel
 * 4. Subscription documents send as "doc" events on the control channel
 * 5. Server creates a subscription channel for each active subscription
 * 6. Data pushes arrive as "subscription:data" events
 *
 * Why not use Apollo Client subscriptions?
 *
 * Apollo expects the graphql-ws or subscriptions-transport-ws protocols.
 * Phoenix/Absinthe uses a different protocol optimized for Elixir's
 * process model. While adapters exist, using Phoenix channels directly
 * provides better integration with Phoenix presence, better error handling,
 * and avoids protocol translation overhead.
 *
 * Reconnection strategy:
 *
 * WebSocket connections are unreliable. Mobile networks, laptop sleep,
 * network switches all cause disconnections. This hook implements
 * exponential backoff reconnection:
 *
 * - First retry: 1 second
 * - Second retry: 2 seconds
 * - Third retry: 4 seconds
 * - Maximum: 30 seconds
 *
 * On reconnection, active subscriptions automatically resubscribe.
 * The server may have processed events during disconnection; clients
 * should handle potential gaps by fetching current state on reconnect.
 *
 * Sequence tracking:
 *
 * Each subscription message includes a sequence number. Clients track
 * the last received sequence; gaps indicate missed messages. On gap
 * detection, the hook triggers a resync callback allowing the component
 * to fetch current state via a query.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { Socket, Channel } from 'phoenix';

// Types

interface SubscriptionOptions<TData, TVariables = Record<string, unknown>> {
  /** GraphQL subscription document */
  query: string;
  /** Variables to pass to the subscription */
  variables?: TVariables;
  /** Called when new data arrives */
  onData?: (data: TData) => void;
  /** Called on subscription error */
  onError?: (error: SubscriptionError) => void;
  /** Called when connection state changes */
  onConnectionChange?: (state: ConnectionState) => void;
  /** Called when sequence gap detected (missed messages) */
  onResyncNeeded?: () => void;
  /** Skip subscription (for conditional subscriptions) */
  skip?: boolean;
  /** Authentication token for socket connection */
  token?: string;
}

interface SubscriptionError {
  message: string;
  code?: string;
  extensions?: Record<string, unknown>;
}

type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error';

interface SubscriptionResult<TData> {
  data: TData | null;
  loading: boolean;
  error: SubscriptionError | null;
  connectionState: ConnectionState;
  /** Manually reconnect the subscription */
  reconnect: () => void;
  /** Pause the subscription without disconnecting */
  pause: () => void;
  /** Resume a paused subscription */
  resume: () => void;
}

interface PhoenixMessage<T = unknown> {
  result?: {
    data?: T;
    errors?: Array<{ message: string; extensions?: Record<string, unknown> }>;
  };
  subscriptionId?: string;
}

// Singleton socket manager to share connection across hooks

class SocketManager {
  private static instance: SocketManager;
  private socket: Socket | null = null;
  private controlChannel: Channel | null = null;
  private subscriptionChannels: Map<string, Channel> = new Map();
  private connectionState: ConnectionState = 'disconnected';
  private listeners: Set<(state: ConnectionState) => void> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;

  static getInstance(): SocketManager {
    if (!SocketManager.instance) {
      SocketManager.instance = new SocketManager();
    }
    return SocketManager.instance;
  }

  connect(endpoint: string, token?: string): Promise<void> {
    if (this.socket?.isConnected()) {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      const params = token ? { token } : {};

      this.socket = new Socket(endpoint, {
        params,
        reconnectAfterMs: (tries) => this.calculateBackoff(tries),
        heartbeatIntervalMs: 30000,
        logger: (kind, msg, data) => {
          if (process.env.NODE_ENV === 'development') {
            console.debug(`[Phoenix ${kind}]`, msg, data);
          }
        },
      });

      this.socket.onOpen(() => {
        this.reconnectAttempts = 0;
        this.setConnectionState('connected');
        this.joinControlChannel().then(resolve).catch(reject);
      });

      this.socket.onClose(() => {
        this.setConnectionState('disconnected');
      });

      this.socket.onError((error) => {
        console.error('[Phoenix Socket Error]', error);
        this.setConnectionState('error');
        this.reconnectAttempts++;

        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          reject(new Error('Max reconnection attempts exceeded'));
        }
      });

      this.setConnectionState('connecting');
      this.socket.connect();
    });
  }

  private calculateBackoff(attempts: number): number {
    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
    const baseDelay = 1000;
    const maxDelay = 30000;
    const delay = Math.min(baseDelay * Math.pow(2, attempts), maxDelay);
    // Add jitter to prevent thundering herd
    return delay + Math.random() * 1000;
  }

  private async joinControlChannel(): Promise<void> {
    if (!this.socket) {
      throw new Error('Socket not initialized');
    }

    this.controlChannel = this.socket.channel('__absinthe__:control', {});

    return new Promise((resolve, reject) => {
      this.controlChannel!
        .join()
        .receive('ok', () => {
          console.debug('[Absinthe] Control channel joined');
          resolve();
        })
        .receive('error', (resp) => {
          console.error('[Absinthe] Control channel join failed', resp);
          reject(new Error('Failed to join control channel'));
        })
        .receive('timeout', () => {
          reject(new Error('Control channel join timeout'));
        });
    });
  }

  subscribe<T>(
    subscriptionId: string,
    query: string,
    variables: Record<string, unknown>,
    onData: (data: T) => void,
    onError: (error: SubscriptionError) => void
  ): () => void {
    if (!this.controlChannel) {
      onError({ message: 'Not connected to server' });
      return () => {};
    }

    // Send subscription document to control channel
    this.controlChannel
      .push('doc', {
        query,
        variables,
      })
      .receive('ok', (response: PhoenixMessage<T>) => {
        if (response.subscriptionId) {
          this.setupSubscriptionChannel(
            subscriptionId,
            response.subscriptionId,
            onData,
            onError
          );
        } else if (response.result?.errors) {
          onError({
            message: response.result.errors[0].message,
            extensions: response.result.errors[0].extensions,
          });
        }
      })
      .receive('error', (error) => {
        onError({ message: 'Subscription failed', extensions: error });
      });

    // Return unsubscribe function
    return () => {
      this.unsubscribe(subscriptionId);
    };
  }

  private setupSubscriptionChannel<T>(
    clientId: string,
    serverId: string,
    onData: (data: T) => void,
    onError: (error: SubscriptionError) => void
  ): void {
    if (!this.socket) return;

    const channel = this.socket.channel(`__absinthe__:doc:${serverId}`, {});

    channel.on('subscription:data', (payload: { result: { data: T } }) => {
      if (payload.result?.data) {
        onData(payload.result.data);
      }
    });

    channel.on('subscription:error', (payload: { errors: Array<{ message: string }> }) => {
      if (payload.errors?.[0]) {
        onError({ message: payload.errors[0].message });
      }
    });

    channel
      .join()
      .receive('ok', () => {
        console.debug(`[Absinthe] Subscription channel ${serverId} joined`);
      })
      .receive('error', (resp) => {
        console.error('[Absinthe] Subscription channel join failed', resp);
        onError({ message: 'Failed to establish subscription' });
      });

    this.subscriptionChannels.set(clientId, channel);
  }

  unsubscribe(subscriptionId: string): void {
    const channel = this.subscriptionChannels.get(subscriptionId);
    if (channel) {
      channel.leave();
      this.subscriptionChannels.delete(subscriptionId);
    }
  }

  disconnect(): void {
    this.subscriptionChannels.forEach((channel) => channel.leave());
    this.subscriptionChannels.clear();
    this.controlChannel?.leave();
    this.socket?.disconnect();
    this.socket = null;
    this.controlChannel = null;
    this.setConnectionState('disconnected');
  }

  private setConnectionState(state: ConnectionState): void {
    this.connectionState = state;
    this.listeners.forEach((listener) => listener(state));
  }

  getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  onConnectionChange(listener: (state: ConnectionState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

// Main hook

export function useSubscription<TData, TVariables = Record<string, unknown>>(
  options: SubscriptionOptions<TData, TVariables>
): SubscriptionResult<TData> {
  const {
    query,
    variables,
    onData,
    onError,
    onConnectionChange,
    onResyncNeeded,
    skip = false,
    token,
  } = options;

  const [data, setData] = useState<TData | null>(null);
  const [loading, setLoading] = useState(!skip);
  const [error, setError] = useState<SubscriptionError | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [isPaused, setIsPaused] = useState(false);

  const subscriptionIdRef = useRef<string>(generateSubscriptionId());
  const lastSequenceRef = useRef<number>(0);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  const socketManager = SocketManager.getInstance();

  // Track sequence numbers for gap detection
  const handleData = useCallback(
    (newData: TData) => {
      const payload = newData as unknown as { sequence?: number };
      if (payload.sequence !== undefined) {
        const expectedSequence = lastSequenceRef.current + 1;
        if (payload.sequence > expectedSequence && lastSequenceRef.current > 0) {
          console.warn(
            `[Subscription] Sequence gap detected: expected ${expectedSequence}, got ${payload.sequence}`
          );
          onResyncNeeded?.();
        }
        lastSequenceRef.current = payload.sequence;
      }

      setData(newData);
      setLoading(false);
      setError(null);
      onData?.(newData);
    },
    [onData, onResyncNeeded]
  );

  const handleError = useCallback(
    (subscriptionError: SubscriptionError) => {
      setError(subscriptionError);
      setLoading(false);
      onError?.(subscriptionError);
    },
    [onError]
  );

  // Subscribe effect
  useEffect(() => {
    if (skip || isPaused) {
      return;
    }

    const endpoint =
      process.env.REACT_APP_WS_ENDPOINT ||
      `ws://${window.location.host}/socket/websocket`;

    setLoading(true);

    socketManager
      .connect(endpoint, token)
      .then(() => {
        unsubscribeRef.current = socketManager.subscribe<TData>(
          subscriptionIdRef.current,
          query,
          variables || {},
          handleData,
          handleError
        );
      })
      .catch((connectionError) => {
        handleError({
          message: connectionError.message || 'Connection failed',
        });
      });

    return () => {
      unsubscribeRef.current?.();
      unsubscribeRef.current = null;
    };
  }, [query, JSON.stringify(variables), skip, isPaused, token, handleData, handleError]);

  // Connection state listener
  useEffect(() => {
    const unsubscribe = socketManager.onConnectionChange((state) => {
      setConnectionState(state);
      onConnectionChange?.(state);

      // Reset sequence on reconnect to allow resync
      if (state === 'connected' && lastSequenceRef.current > 0) {
        lastSequenceRef.current = 0;
        onResyncNeeded?.();
      }
    });

    setConnectionState(socketManager.getConnectionState());

    return unsubscribe;
  }, [onConnectionChange, onResyncNeeded]);

  const reconnect = useCallback(() => {
    unsubscribeRef.current?.();
    lastSequenceRef.current = 0;
    subscriptionIdRef.current = generateSubscriptionId();
    setLoading(true);
    setError(null);
    setIsPaused(false);
  }, []);

  const pause = useCallback(() => {
    setIsPaused(true);
    unsubscribeRef.current?.();
  }, []);

  const resume = useCallback(() => {
    setIsPaused(false);
  }, []);

  return {
    data,
    loading,
    error,
    connectionState,
    reconnect,
    pause,
    resume,
  };
}

// Specialized hooks for common subscription patterns

/**
 * Subscribe to real time metric updates
 */
export function useMetricsSubscription(options?: {
  onUpdate?: (metrics: MetricUpdate) => void;
  skip?: boolean;
}) {
  return useSubscription<{ metricsUpdated: MetricUpdate }>({
    query: `
      subscription MetricsUpdated {
        metricsUpdated {
          metric
          value
          previousValue
          changePercent
          trend
          timestamp
          sequence
        }
      }
    `,
    onData: (data) => options?.onUpdate?.(data.metricsUpdated),
    skip: options?.skip,
  });
}

/**
 * Subscribe to specific metric changes
 */
export function useMetricSubscription(
  metricName: MetricName,
  options?: {
    onUpdate?: (update: MetricUpdate) => void;
    skip?: boolean;
  }
) {
  return useSubscription<{ metricChanged: MetricUpdate }>({
    query: `
      subscription MetricChanged($metricName: MetricNameEnum!) {
        metricChanged(metricName: $metricName) {
          metric
          value
          changePercent
          trend
          timestamp
          sequence
        }
      }
    `,
    variables: { metricName },
    onData: (data) => options?.onUpdate?.(data.metricChanged),
    skip: options?.skip,
  });
}

/**
 * Subscribe to threshold alerts
 */
export function useThresholdAlerts(
  thresholds: ThresholdConfig,
  onAlert: (alert: ThresholdAlert) => void
) {
  return useSubscription<{ thresholdAlert: ThresholdAlert }>({
    query: `
      subscription ThresholdAlerts($thresholds: ThresholdInput!) {
        thresholdAlert(thresholds: $thresholds) {
          metric
          threshold
          currentValue
          direction
          triggeredAt
          message
        }
      }
    `,
    variables: { thresholds },
    onData: (data) => {
      if (data.thresholdAlert) {
        onAlert(data.thresholdAlert);
      }
    },
  });
}

// Type definitions

interface MetricUpdate {
  metric: MetricName;
  value: string;
  previousValue?: string;
  changePercent?: number;
  trend: 'up' | 'down' | 'flat';
  timestamp: string;
  sequence: number;
}

type MetricName =
  | 'ACTIVE_USERS'
  | 'TOTAL_USERS'
  | 'NEW_USERS_TODAY'
  | 'ORDERS_TODAY'
  | 'REVENUE_TODAY'
  | 'ORDERS_PER_MINUTE';

interface ThresholdConfig {
  activeUsers?: number;
  ordersPerMinute?: number;
  revenueToday?: number;
}

interface ThresholdAlert {
  metric: MetricName;
  threshold: number;
  currentValue: number;
  direction: 'above' | 'below';
  triggeredAt: string;
  message?: string;
}

// Utility functions

function generateSubscriptionId(): string {
  return `sub_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

export type {
  SubscriptionOptions,
  SubscriptionResult,
  SubscriptionError,
  ConnectionState,
  MetricUpdate,
  MetricName,
  ThresholdConfig,
  ThresholdAlert,
};
