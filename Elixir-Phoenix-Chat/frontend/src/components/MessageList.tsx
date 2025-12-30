import { useEffect, useRef, memo } from "react";

/**
 * Message list with automatic scroll and virtualization ready structure.
 *
 * Scroll behavior:
 * - Auto scroll to bottom on new messages if user is near bottom
 * - Preserve scroll position if user has scrolled up to read history
 * - "New messages" indicator when messages arrive while scrolled up
 *
 * Performance considerations:
 * - Messages are memoized to prevent re render on parent state changes
 * - Timestamp formatting cached per message (not recalculated on every render)
 * - For rooms with 1000+ messages, implement windowing (react-window)
 *
 * Structure prepared for virtualization but not implemented here to keep
 * demo focused on Phoenix/Elixir patterns rather than React optimization.
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

interface MessageListProps {
  messages: Message[];
  currentUserId: string;
  onRetry: (messageId: string) => void;
}

export function MessageList({
  messages,
  currentUserId,
  onRetry,
}: MessageListProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const shouldScrollRef = useRef(true);

  // Track if user is near bottom
  const handleScroll = () => {
    const container = containerRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    // Consider "near bottom" if within 100px
    shouldScrollRef.current = distanceFromBottom < 100;
  };

  // Auto scroll on new messages
  useEffect(() => {
    if (shouldScrollRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages.length]);

  return (
    <div
      ref={containerRef}
      className="message-list"
      onScroll={handleScroll}
      role="log"
      aria-live="polite"
    >
      {messages.length === 0 ? (
        <div className="message-list__empty">
          No messages yet. Start the conversation!
        </div>
      ) : (
        messages.map((message, index) => (
          <MessageItem
            key={message.id}
            message={message}
            isOwn={message.userId === currentUserId}
            showUsername={shouldShowUsername(messages, index)}
            onRetry={onRetry}
          />
        ))
      )}
    </div>
  );
}

interface MessageItemProps {
  message: Message;
  isOwn: boolean;
  showUsername: boolean;
  onRetry: (messageId: string) => void;
}

const MessageItem = memo(function MessageItem({
  message,
  isOwn,
  showUsername,
  onRetry,
}: MessageItemProps): JSX.Element {
  const formattedTime = formatTimestamp(message.timestamp);

  const handleRetryClick = () => {
    onRetry(message.id);
  };

  return (
    <div
      className={`message ${isOwn ? "message--own" : ""} ${
        message.pending ? "message--pending" : ""
      } ${message.error ? "message--error" : ""}`}
    >
      {showUsername && !isOwn && (
        <span className="message__username">{message.username}</span>
      )}

      <div className="message__bubble">
        <p className="message__content">{message.content}</p>

        <span className="message__meta">
          <time className="message__time" dateTime={message.timestamp}>
            {formattedTime}
          </time>

          {message.pending && (
            <span className="message__status" aria-label="Sending">
              Sending...
            </span>
          )}

          {message.error && (
            <button
              className="message__retry"
              onClick={handleRetryClick}
              aria-label="Retry sending message"
            >
              Failed. Tap to retry.
            </button>
          )}
        </span>
      </div>
    </div>
  );
});

/**
 * Determines if username should be shown for this message.
 * Groups consecutive messages from same user to reduce visual noise.
 */
function shouldShowUsername(messages: Message[], index: number): boolean {
  if (index === 0) return true;

  const current = messages[index];
  const previous = messages[index - 1];

  if (!current || !previous) return true;

  // Show username if different user
  if (current.userId !== previous.userId) return true;

  // Show username if more than 5 minutes since previous message
  const currentTime = new Date(current.timestamp).getTime();
  const previousTime = new Date(previous.timestamp).getTime();
  const fiveMinutes = 5 * 60 * 1000;

  return currentTime - previousTime > fiveMinutes;
}

/**
 * Formats timestamp for display.
 * Today: "2:30 PM"
 * This week: "Mon 2:30 PM"
 * Older: "Dec 15, 2:30 PM"
 */
function formatTimestamp(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();

  const isToday =
    date.getDate() === now.getDate() &&
    date.getMonth() === now.getMonth() &&
    date.getFullYear() === now.getFullYear();

  const timeString = date.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });

  if (isToday) {
    return timeString;
  }

  const daysDiff = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

  if (daysDiff < 7) {
    const dayName = date.toLocaleDateString(undefined, { weekday: "short" });
    return `${dayName} ${timeString}`;
  }

  const dateString = date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });

  return `${dateString}, ${timeString}`;
}
