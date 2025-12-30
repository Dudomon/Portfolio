import { useEffect, useState, useCallback, useRef } from "react";
import { Socket, Channel, Presence } from "phoenix";

/**
 * React hook for Phoenix Channel integration.
 *
 * Design decisions:
 *
 * 1. Single socket, multiple channels: One WebSocket connection handles all
 *    room subscriptions. Creating new socket per room would waste TCP connections
 *    and bypass Phoenix's built in multiplexing.
 *
 * 2. Automatic reconnection: Phoenix socket handles reconnection with exponential
 *    backoff. Hook re subscribes to channels after reconnect. User sees brief
 *    "reconnecting" state, not manual refresh required.
 *
 * 3. Presence as separate concern: usePresence hook composes with useChannel.
 *    Separation allows using channel without presence overhead for non chat
 *    use cases (notifications, live updates).
 *
 * 4. Cleanup on unmount: Channel leaves and event handlers removed when component
 *    unmounts. Prevents memory leaks from orphaned subscriptions.
 */

// Singleton socket instance shared across all hooks
let socketInstance: Socket | null = null;

function getSocket(token: string): Socket {
  if (!socketInstance) {
    socketInstance = new Socket("/socket", {
      params: { token },
      reconnectAfterMs: (tries) => [1000, 2000, 5000, 10000][tries - 1] || 10000,
    });
    socketInstance.connect();
  }
  return socketInstance;
}

export function disconnectSocket(): void {
  if (socketInstance) {
    socketInstance.disconnect();
    socketInstance = null;
  }
}

interface UseChannelOptions {
  token: string;
  topic: string;
  onJoin?: (response: unknown) => void;
  onError?: (error: unknown) => void;
  onClose?: () => void;
}

interface UseChannelReturn {
  channel: Channel | null;
  connected: boolean;
  error: string | null;
  push: (event: string, payload: object) => Promise<unknown>;
  on: (event: string, callback: (payload: unknown) => void) => void;
  off: (event: string) => void;
}

export function useChannel({
  token,
  topic,
  onJoin,
  onError,
  onClose,
}: UseChannelOptions): UseChannelReturn {
  const [channel, setChannel] = useState<Channel | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track event handlers for cleanup
  const handlersRef = useRef<Map<string, (payload: unknown) => void>>(new Map());

  useEffect(() => {
    const socket = getSocket(token);
    const chan = socket.channel(topic, {});

    chan
      .join()
      .receive("ok", (response) => {
        setConnected(true);
        setError(null);
        onJoin?.(response);
      })
      .receive("error", (resp) => {
        setError(resp.reason || "Failed to join channel");
        onError?.(resp);
      })
      .receive("timeout", () => {
        setError("Connection timeout");
      });

    chan.onClose(() => {
      setConnected(false);
      onClose?.();
    });

    chan.onError(() => {
      setConnected(false);
      setError("Channel error");
    });

    setChannel(chan);

    return () => {
      // Remove all event handlers
      handlersRef.current.forEach((_, event) => {
        chan.off(event);
      });
      handlersRef.current.clear();

      // Leave channel
      chan.leave();
      setChannel(null);
      setConnected(false);
    };
  }, [token, topic]);

  const push = useCallback(
    (event: string, payload: object): Promise<unknown> => {
      return new Promise((resolve, reject) => {
        if (!channel) {
          reject(new Error("Channel not connected"));
          return;
        }

        channel
          .push(event, payload)
          .receive("ok", resolve)
          .receive("error", reject)
          .receive("timeout", () => reject(new Error("Request timeout")));
      });
    },
    [channel]
  );

  const on = useCallback(
    (event: string, callback: (payload: unknown) => void) => {
      if (!channel) return;

      // Remove existing handler for this event
      if (handlersRef.current.has(event)) {
        channel.off(event);
      }

      channel.on(event, callback);
      handlersRef.current.set(event, callback);
    },
    [channel]
  );

  const off = useCallback(
    (event: string) => {
      if (!channel) return;

      channel.off(event);
      handlersRef.current.delete(event);
    },
    [channel]
  );

  return { channel, connected, error, push, on, off };
}

/**
 * Hook for Phoenix Presence integration.
 * Tracks online users in a channel with automatic sync.
 */
interface PresenceUser {
  id: string;
  username: string;
  onlineAt: number;
  status: string;
}

interface UsePresenceReturn {
  users: PresenceUser[];
}

export function usePresence(channel: Channel | null): UsePresenceReturn {
  const [users, setUsers] = useState<PresenceUser[]>([]);
  const presenceRef = useRef<Presence | null>(null);

  useEffect(() => {
    if (!channel) return;

    const presence = new Presence(channel);
    presenceRef.current = presence;

    const syncUsers = () => {
      const list: PresenceUser[] = [];

      presence.list((id, { metas }) => {
        // Take first meta (user might have multiple tabs)
        const meta = metas[0];
        if (meta) {
          list.push({
            id,
            username: meta.username,
            onlineAt: meta.online_at,
            status: meta.status || "active",
          });
        }
      });

      setUsers(list);
    };

    presence.onSync(syncUsers);

    return () => {
      presenceRef.current = null;
    };
  }, [channel]);

  return { users };
}
