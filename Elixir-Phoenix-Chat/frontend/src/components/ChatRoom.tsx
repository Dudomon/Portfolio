import { useState, useEffect, useRef, useCallback } from "react";
import { useChannel, usePresence } from "../hooks/useChannel";
import { MessageList } from "./MessageList";
import { UserList } from "./UserList";

/**
 * Main chat room component.
 *
 * State management approach:
 * - Messages stored in local state, not global store. Each room instance
 *   manages its own messages. Global store would add complexity without
 *   benefit since users view one room at a time.
 *
 * - Optimistic updates: New message appears immediately in UI before server
 *   confirmation. If server rejects (validation error), message is removed
 *   and error shown. This provides instant feedback for 99%+ of messages.
 *
 * - Typing indicator uses debounce: Keystrokes trigger "typing" event,
 *   but we debounce to avoid flooding server. Typing stops 3 seconds after
 *   last keystroke (server side timeout).
 */

interface Message {
  id: string;
  userId: string;
  username: string;
  content: string;
  timestamp: string;
  pending?: boolean;
  error?: boolean;
}

interface ChatRoomProps {
  roomName: string;
  token: string;
  currentUserId: string;
  currentUsername: string;
}

export function ChatRoom({
  roomName,
  token,
  currentUserId,
  currentUsername,
}: ChatRoomProps): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [typingUsers, setTypingUsers] = useState<string[]>([]);

  const inputRef = useRef<HTMLInputElement>(null);
  const typingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { channel, connected, error, push, on } = useChannel({
    token,
    topic: `room:${roomName}`,
    onJoin: (response) => {
      const resp = response as { messages: Message[] };
      setMessages(resp.messages || []);
    },
  });

  const { users } = usePresence(channel);

  // Subscribe to new messages
  useEffect(() => {
    if (!channel) return;

    on("new_message", (payload) => {
      const message = payload as Message;

      setMessages((prev) => {
        // Check if this is confirmation of our optimistic message
        const existingIndex = prev.findIndex(
          (m) => m.pending && m.content === message.content && m.userId === message.userId
        );

        if (existingIndex >= 0) {
          // Replace pending message with confirmed one
          const updated = [...prev];
          updated[existingIndex] = message;
          return updated;
        }

        // New message from another user
        return [...prev, message];
      });
    });

    on("typing", (payload) => {
      const { userId, username, isTyping } = payload as {
        userId: string;
        username: string;
        isTyping: boolean;
      };

      // Ignore own typing events
      if (userId === currentUserId) return;

      setTypingUsers((prev) => {
        if (isTyping && !prev.includes(username)) {
          return [...prev, username];
        }
        if (!isTyping) {
          return prev.filter((u) => u !== username);
        }
        return prev;
      });
    });
  }, [channel, on, currentUserId]);

  const sendMessage = useCallback(async () => {
    const content = inputValue.trim();
    if (!content || !connected) return;

    // Clear input immediately for responsiveness
    setInputValue("");

    // Optimistic update: add message before server confirmation
    const optimisticMessage: Message = {
      id: `pending-${Date.now()}`,
      userId: currentUserId,
      username: currentUsername,
      content,
      timestamp: new Date().toISOString(),
      pending: true,
    };

    setMessages((prev) => [...prev, optimisticMessage]);

    try {
      await push("new_message", { content });
      // Stop typing indicator after sending
      push("stop_typing", {});
    } catch (err) {
      // Mark message as failed
      setMessages((prev) =>
        prev.map((m) =>
          m.id === optimisticMessage.id ? { ...m, pending: false, error: true } : m
        )
      );
    }
  }, [inputValue, connected, currentUserId, currentUsername, push]);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setInputValue(e.target.value);

      // Send typing indicator (debounced)
      if (connected) {
        push("typing", {});

        // Clear existing timeout
        if (typingTimeoutRef.current) {
          clearTimeout(typingTimeoutRef.current);
        }

        // Stop typing after 2 seconds of no input
        typingTimeoutRef.current = setTimeout(() => {
          push("stop_typing", {});
        }, 2000);
      }
    },
    [connected, push]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage]
  );

  const retryMessage = useCallback(
    async (messageId: string) => {
      const message = messages.find((m) => m.id === messageId);
      if (!message) return;

      // Reset error state
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, error: false, pending: true } : m))
      );

      try {
        await push("new_message", { content: message.content });
      } catch {
        setMessages((prev) =>
          prev.map((m) => (m.id === messageId ? { ...m, pending: false, error: true } : m))
        );
      }
    },
    [messages, push]
  );

  return (
    <div className="chat-room">
      <header className="chat-room__header">
        <h1 className="chat-room__title">{roomName}</h1>
        <span className="chat-room__status">
          {connected ? (
            <span className="status-indicator status-indicator--connected">Connected</span>
          ) : error ? (
            <span className="status-indicator status-indicator--error">{error}</span>
          ) : (
            <span className="status-indicator status-indicator--connecting">Connecting...</span>
          )}
        </span>
      </header>

      <div className="chat-room__content">
        <main className="chat-room__messages">
          <MessageList
            messages={messages}
            currentUserId={currentUserId}
            onRetry={retryMessage}
          />

          {typingUsers.length > 0 && (
            <div className="typing-indicator">
              {formatTypingUsers(typingUsers)}
            </div>
          )}
        </main>

        <aside className="chat-room__users">
          <UserList users={users} currentUserId={currentUserId} />
        </aside>
      </div>

      <footer className="chat-room__input">
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={connected ? "Type a message..." : "Connecting..."}
          disabled={!connected}
          maxLength={2000}
          aria-label="Message input"
        />
        <button
          onClick={sendMessage}
          disabled={!connected || !inputValue.trim()}
          aria-label="Send message"
        >
          Send
        </button>
      </footer>
    </div>
  );
}

function formatTypingUsers(users: string[]): string {
  if (users.length === 1) {
    return `${users[0]} is typing...`;
  }
  if (users.length === 2) {
    return `${users[0]} and ${users[1]} are typing...`;
  }
  return `${users[0]} and ${users.length - 1} others are typing...`;
}
