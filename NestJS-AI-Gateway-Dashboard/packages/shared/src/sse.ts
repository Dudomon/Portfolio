/**
 * Minimal Server-Sent Events parser following the WHATWG event-stream rules
 * (https://html.spec.whatwg.org/multipage/server-sent-events.html).
 *
 * Used by the SDK to read gateway streams and by the gateway to read upstream
 * provider streams, so both sides share one tested implementation.
 */

export interface SseEvent {
  /** Event name; defaults to "message" per the spec. */
  event: string;
  data: string;
  id?: string;
}

/**
 * Incremental line-oriented parser. Feed it arbitrary text chunks (which may
 * split lines, or even CRLF pairs, at any point) and it yields complete events.
 */
export class SseParser {
  private buffer = '';
  private dataLines: string[] = [];
  private eventName = '';
  private lastEventId: string | undefined;
  private pendingCarriageReturn = false;

  push(chunk: string): SseEvent[] {
    let text = chunk;
    // A chunk boundary may fall between "\r" and "\n"; the "\r" already ended
    // the line, so the dangling "\n" must not produce a second, empty line.
    if (this.pendingCarriageReturn && text.startsWith('\n')) text = text.slice(1);
    this.pendingCarriageReturn = text.endsWith('\r');
    this.buffer += text;

    const events: SseEvent[] = [];
    const lines = this.buffer.split(/\r\n|\r|\n/);
    // The last element is an incomplete line (possibly empty); keep it.
    this.buffer = lines.pop() ?? '';
    for (const line of lines) {
      const event = this.processLine(line);
      if (event) events.push(event);
    }
    return events;
  }

  /** Flushes a trailing event when the stream ends without a blank line. */
  end(): SseEvent[] {
    const events: SseEvent[] = [];
    if (this.buffer.length > 0) {
      const event = this.processLine(this.buffer);
      if (event) events.push(event);
      this.buffer = '';
    }
    const trailing = this.dispatch();
    if (trailing) events.push(trailing);
    return events;
  }

  private processLine(line: string): SseEvent | undefined {
    if (line === '') return this.dispatch();
    if (line.startsWith(':')) return undefined;

    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);

    switch (field) {
      case 'data':
        this.dataLines.push(value);
        break;
      case 'event':
        this.eventName = value;
        break;
      case 'id':
        if (!value.includes('\0')) this.lastEventId = value;
        break;
      default:
        // "retry" and unknown fields carry no meaning for request/response streams.
        break;
    }
    return undefined;
  }

  private dispatch(): SseEvent | undefined {
    if (this.dataLines.length === 0) {
      this.eventName = '';
      return undefined;
    }
    const event: SseEvent = {
      event: this.eventName || 'message',
      data: this.dataLines.join('\n'),
      ...(this.lastEventId !== undefined ? { id: this.lastEventId } : {}),
    };
    this.dataLines = [];
    this.eventName = '';
    return event;
  }
}

/**
 * Reads a byte stream and yields SSE events. Uses the reader API rather than
 * async iteration of ReadableStream because Safari does not implement the
 * latter; the reader is canceled if the consumer stops early.
 */
export async function* readSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseEvent, void, undefined> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  let completed = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }
    yield* parser.push(decoder.decode());
    yield* parser.end();
    completed = true;
  } finally {
    if (!completed) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
