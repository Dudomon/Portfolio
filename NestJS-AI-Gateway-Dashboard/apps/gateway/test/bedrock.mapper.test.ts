import type { ConverseStreamOutput } from '@aws-sdk/client-bedrock-runtime';
import { describe, expect, it } from 'vitest';
import { ConverseStreamTranslator, toConverseInput } from '../src/providers/bedrock/bedrock.mapper';
import { UpstreamError } from '../src/providers/upstream-error';

describe('toConverseInput', () => {
  it('maps system prompts, inference config, tool results and tool choice', () => {
    const input = toConverseInput(
      {
        model: 'm',
        messages: [
          { role: 'system', content: 'Be brief.' },
          { role: 'user', content: 'Weather?' },
          {
            role: 'assistant',
            content: null,
            tool_calls: [{ id: 't1', type: 'function', function: { name: 'w', arguments: '{}' } }],
          },
          { role: 'tool', tool_call_id: 't1', content: 'sunny' },
        ],
        max_tokens: 100,
        temperature: 0.2,
        tools: [{ type: 'function', function: { name: 'w' } }],
        tool_choice: { type: 'function', function: { name: 'w' } },
      },
      4_096,
    );

    expect(input.system).toEqual([{ text: 'Be brief.' }]);
    expect(input.inferenceConfig).toEqual({ maxTokens: 100, temperature: 0.2 });
    expect(input.messages?.[2]).toEqual({
      role: 'user',
      content: [{ toolResult: { toolUseId: 't1', content: [{ text: 'sunny' }] } }],
    });
    expect(input.toolConfig?.toolChoice).toEqual({ tool: { name: 'w' } });
  });
});

describe('ConverseStreamTranslator', () => {
  it('translates text, tool use, stop reason and trailing usage metadata', () => {
    const translator = new ConverseStreamTranslator('id', 'm', 1);
    const events: ConverseStreamOutput[] = [
      { messageStart: { role: 'assistant' } },
      { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'Hi' } } },
      {
        contentBlockStart: {
          contentBlockIndex: 1,
          start: { toolUse: { toolUseId: 'tu', name: 'w' } },
        },
      },
      { contentBlockDelta: { contentBlockIndex: 1, delta: { toolUse: { input: '{"a":1}' } } } },
      { messageStop: { stopReason: 'tool_use' } },
      {
        metadata: {
          usage: { inputTokens: 7, outputTokens: 3, totalTokens: 10 },
          metrics: { latencyMs: 5 },
        },
      },
    ];

    const chunks = events.flatMap((event) => translator.translate(event));

    expect(chunks.map((c) => c.choices[0]?.delta.content).filter(Boolean)).toEqual(['Hi']);
    expect(chunks[2]?.choices[0]?.delta.tool_calls?.[0]).toMatchObject({ index: 0, id: 'tu' });
    expect(chunks[3]?.choices[0]?.delta.tool_calls?.[0]?.function?.arguments).toBe('{"a":1}');
    expect(chunks[4]?.choices[0]?.finish_reason).toBe('tool_calls');
    expect(chunks[5]?.usage).toEqual({ prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 });
    expect(translator.finished).toBe(true);
  });

  it('classifies in-stream throttling as a rate limit', () => {
    const translator = new ConverseStreamTranslator('id', 'm', 1);
    let caught: unknown;
    try {
      translator.translate({
        throttlingException: {
          name: 'ThrottlingException',
          message: 'slow down',
          $fault: 'client',
          $metadata: {},
        },
      });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(UpstreamError);
    expect((caught as UpstreamError).kind).toBe('rate_limit');
  });
});
