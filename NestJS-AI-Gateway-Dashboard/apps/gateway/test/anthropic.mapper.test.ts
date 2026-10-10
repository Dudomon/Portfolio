import { describe, expect, it } from 'vitest';
import type { ChatCompletionChunk } from '@aigw/shared';
import {
  AnthropicStreamTranslator,
  fromAnthropicResponse,
  toAnthropicRequest,
} from '../src/providers/anthropic/anthropic.mapper';
import type { AnthropicStreamEvent } from '../src/providers/anthropic/anthropic.types';
import { UpstreamError } from '../src/providers/upstream-error';

describe('toAnthropicRequest', () => {
  it('moves system prompts out of band and applies the default max_tokens', () => {
    const result = toAnthropicRequest(
      {
        model: 'claude-test',
        messages: [
          { role: 'system', content: 'Be terse.' },
          { role: 'developer', content: 'Use metric units.' },
          { role: 'user', content: 'Hi' },
        ],
        temperature: 1.6,
        stop: 'END',
      },
      2_048,
    );

    expect(result.system).toBe('Be terse.\n\nUse metric units.');
    expect(result.max_tokens).toBe(2_048);
    expect(result.temperature).toBe(1);
    expect(result.stop_sequences).toEqual(['END']);
    expect(result.messages).toEqual([{ role: 'user', content: [{ type: 'text', text: 'Hi' }] }]);
  });

  it('translates tool calls and merges consecutive tool results into one user turn', () => {
    const result = toAnthropicRequest(
      {
        model: 'claude-test',
        messages: [
          { role: 'user', content: 'Weather in Paris and Rome?' },
          {
            role: 'assistant',
            content: null,
            tool_calls: [
              {
                id: 'call_1',
                type: 'function',
                function: { name: 'weather', arguments: '{"city":"Paris"}' },
              },
              {
                id: 'call_2',
                type: 'function',
                function: { name: 'weather', arguments: '{"city":"Rome"}' },
              },
            ],
          },
          { role: 'tool', tool_call_id: 'call_1', content: '18C' },
          { role: 'tool', tool_call_id: 'call_2', content: '24C' },
        ],
        tools: [
          { type: 'function', function: { name: 'weather', parameters: { type: 'object' } } },
        ],
        tool_choice: 'required',
      },
      1_024,
    );

    expect(result.messages).toHaveLength(3);
    expect(result.messages[1]).toEqual({
      role: 'assistant',
      content: [
        { type: 'tool_use', id: 'call_1', name: 'weather', input: { city: 'Paris' } },
        { type: 'tool_use', id: 'call_2', name: 'weather', input: { city: 'Rome' } },
      ],
    });
    expect(result.messages[2]).toEqual({
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 'call_1', content: '18C' },
        { type: 'tool_result', tool_use_id: 'call_2', content: '24C' },
      ],
    });
    expect(result.tools).toEqual([{ name: 'weather', input_schema: { type: 'object' } }]);
    expect(result.tool_choice).toEqual({ type: 'any' });
  });

  it('converts data URL images to base64 sources', () => {
    const result = toAnthropicRequest(
      {
        model: 'm',
        messages: [
          {
            role: 'user',
            content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } }],
          },
        ],
      },
      1,
    );
    expect(result.messages[0]?.content[0]).toEqual({
      type: 'image',
      source: { type: 'base64', media_type: 'image/png', data: 'AAAA' },
    });
  });
});

describe('fromAnthropicResponse', () => {
  it('maps text, tool_use, stop reason and cache-aware usage', () => {
    const completion = fromAnthropicResponse(
      {
        id: 'msg_1',
        type: 'message',
        role: 'assistant',
        model: 'claude-test',
        content: [
          { type: 'text', text: 'Checking.' },
          { type: 'tool_use', id: 'toolu_1', name: 'weather', input: { city: 'Paris' } },
        ],
        stop_reason: 'tool_use',
        usage: { input_tokens: 10, cache_read_input_tokens: 90, output_tokens: 7 },
      },
      123,
    );

    expect(completion.choices[0]).toEqual({
      index: 0,
      message: {
        role: 'assistant',
        content: 'Checking.',
        tool_calls: [
          {
            id: 'toolu_1',
            type: 'function',
            function: { name: 'weather', arguments: '{"city":"Paris"}' },
          },
        ],
      },
      finish_reason: 'tool_calls',
    });
    expect(completion.usage).toEqual({
      prompt_tokens: 100,
      completion_tokens: 7,
      total_tokens: 107,
    });
  });
});

describe('AnthropicStreamTranslator', () => {
  const events: AnthropicStreamEvent[] = [
    {
      type: 'message_start',
      message: { id: 'msg_1', model: 'claude-test', usage: { input_tokens: 25, output_tokens: 1 } },
    },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'ping' },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Let me check' } },
    { type: 'content_block_stop', index: 0 },
    {
      type: 'content_block_start',
      index: 1,
      content_block: { type: 'tool_use', id: 'toolu_1', name: 'weather' },
    },
    {
      type: 'content_block_delta',
      index: 1,
      delta: { type: 'input_json_delta', partial_json: '' },
    },
    {
      type: 'content_block_delta',
      index: 1,
      delta: { type: 'input_json_delta', partial_json: '{"city":' },
    },
    {
      type: 'content_block_delta',
      index: 1,
      delta: { type: 'input_json_delta', partial_json: '"Paris"}' },
    },
    { type: 'content_block_stop', index: 1 },
    { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 31 } },
    { type: 'message_stop' },
  ];

  function translateAll(): ChatCompletionChunk[] {
    const translator = new AnthropicStreamTranslator('claude-test', 1);
    return events.flatMap((event) => translator.translate(event));
  }

  it('produces an OpenAI-shaped chunk sequence', () => {
    const chunks = translateAll();
    const deltas = chunks.map((c) => c.choices[0]?.delta);

    expect(deltas[0]).toEqual({ role: 'assistant', content: '' });
    expect(deltas[1]).toEqual({ content: 'Let me check' });
    // Anthropic block index 1 becomes OpenAI tool_call index 0.
    expect(deltas[2]).toEqual({
      tool_calls: [
        { index: 0, id: 'toolu_1', type: 'function', function: { name: 'weather', arguments: '' } },
      ],
    });
    const args = chunks
      .flatMap((c) => c.choices[0]?.delta.tool_calls ?? [])
      .map((t) => t.function?.arguments ?? '')
      .join('');
    expect(JSON.parse(args)).toEqual({ city: 'Paris' });
    expect(chunks.every((c) => c.id === 'msg_1')).toBe(true);
  });

  it('emits the finish reason, then a usage-only chunk combining start and delta usage', () => {
    const chunks = translateAll();
    const finish = chunks.at(-2);
    const usage = chunks.at(-1);

    expect(finish?.choices[0]?.finish_reason).toBe('tool_calls');
    expect(usage?.choices).toEqual([]);
    expect(usage?.usage).toEqual({ prompt_tokens: 25, completion_tokens: 31, total_tokens: 56 });
  });

  it('reports completion only after message_stop', () => {
    const translator = new AnthropicStreamTranslator('m', 1);
    for (const event of events.slice(0, -1)) translator.translate(event);
    expect(translator.finished).toBe(false);
    translator.translate({ type: 'message_stop' });
    expect(translator.finished).toBe(true);
  });

  it('turns in-stream overload errors into retryable upstream errors', () => {
    const translator = new AnthropicStreamTranslator('m', 1);
    let caught: unknown;
    try {
      translator.translate({
        type: 'error',
        error: { type: 'overloaded_error', message: 'Overloaded' },
      });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(UpstreamError);
    expect((caught as UpstreamError).kind).toBe('server');
    expect((caught as UpstreamError).isProviderFault).toBe(true);
  });
});
