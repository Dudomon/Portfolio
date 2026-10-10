import { describe, expect, it } from 'vitest';
import { SseParser, readSseStream } from '../src/sse';

function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>) {
  const events = [];
  for await (const event of readSseStream(stream)) events.push(event);
  return events;
}

describe('SseParser', () => {
  it('parses named events with multi-line data', () => {
    const parser = new SseParser();
    const events = parser.push('event: delta\ndata: line one\ndata: line two\n\n');
    expect(events).toEqual([{ event: 'delta', data: 'line one\nline two' }]);
  });

  it('ignores comments and unknown fields', () => {
    const parser = new SseParser();
    expect(parser.push(': keep-alive\nretry: 100\nfoo: bar\ndata: x\n\n')).toEqual([
      { event: 'message', data: 'x' },
    ]);
  });

  it('handles CRLF split across chunks without emitting a phantom event', () => {
    const parser = new SseParser();
    const events = [
      ...parser.push('data: a\r'),
      ...parser.push('\n\r'),
      ...parser.push('\ndata: b\r\n\r\n'),
    ];
    expect(events.map((e) => e.data)).toEqual(['a', 'b']);
  });

  it('only strips a single leading space from values', () => {
    const parser = new SseParser();
    expect(parser.push('data:  padded\n\n')[0]?.data).toBe(' padded');
  });

  it('tracks the last event id', () => {
    const parser = new SseParser();
    expect(parser.push('id: 7\ndata: x\n\n')).toEqual([{ event: 'message', data: 'x', id: '7' }]);
  });
});

describe('readSseStream', () => {
  it('reassembles events split at arbitrary byte boundaries', async () => {
    const events = await collect(streamOf('da', 'ta: {"a":', '1}\n', '\ndata: [DONE]\n\n'));
    expect(events.map((e) => e.data)).toEqual(['{"a":1}', '[DONE]']);
  });

  it('decodes multi-byte UTF-8 characters split across chunks', async () => {
    const bytes = new TextEncoder().encode('data: héllo ✓\n\n');
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
        controller.close();
      },
    });
    expect((await collect(stream))[0]?.data).toBe('héllo ✓');
  });

  it('flushes a trailing event without a terminating blank line', async () => {
    expect(await collect(streamOf('data: tail'))).toEqual([{ event: 'message', data: 'tail' }]);
  });

  it('cancels the underlying stream when the consumer stops early', async () => {
    let canceled = false;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.enqueue(new TextEncoder().encode('data: tick\n\n'));
      },
      cancel() {
        canceled = true;
      },
    });
    for await (const event of readSseStream(stream)) {
      expect(event.data).toBe('tick');
      break;
    }
    expect(canceled).toBe(true);
  });
});
