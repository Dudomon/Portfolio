import {
  contentToText,
  type ChatCompletion,
  type ChatCompletionChunk,
  type ChatCompletionRequest,
  type ChatMessage,
  type ContentPart,
  type FinishReason,
  type ToolCall,
  type Usage,
} from '@aigw/shared';
import { ApiError } from '../../common/api-error';
import { UpstreamError } from '../upstream-error';
import type {
  AnthropicContentBlock,
  AnthropicMessage,
  AnthropicRequest,
  AnthropicResponse,
  AnthropicStopReason,
  AnthropicStreamEvent,
  AnthropicToolChoice,
  AnthropicUsage,
} from './anthropic.types';

/** Translates an OpenAI-format request into an Anthropic Messages request. */
export function toAnthropicRequest(
  request: ChatCompletionRequest,
  defaultMaxTokens: number,
): AnthropicRequest {
  // Anthropic takes system prompts out-of-band rather than as messages.
  const system = request.messages
    .flatMap((m) =>
      m.role === 'system' || m.role === 'developer' ? [contentToText(m.content)] : [],
    )
    .join('\n\n');

  const result: AnthropicRequest = {
    model: request.model,
    max_tokens: request.max_completion_tokens ?? request.max_tokens ?? defaultMaxTokens,
    messages: toAnthropicMessages(request.messages),
  };
  if (system) result.system = system;
  // OpenAI allows temperature up to 2; Anthropic caps at 1. Clamping keeps
  // requests valid across fallbacks at the cost of reduced range.
  if (request.temperature !== undefined) result.temperature = Math.min(request.temperature, 1);
  if (request.top_p !== undefined) result.top_p = request.top_p;
  if (request.stop !== undefined) {
    result.stop_sequences = typeof request.stop === 'string' ? [request.stop] : request.stop;
  }
  if (request.user) result.metadata = { user_id: request.user };
  if (request.tools?.length) {
    result.tools = request.tools.map((tool) => ({
      name: tool.function.name,
      ...(tool.function.description ? { description: tool.function.description } : {}),
      input_schema: tool.function.parameters ?? { type: 'object', properties: {} },
    }));
    if (request.tool_choice !== undefined) result.tool_choice = toToolChoice(request.tool_choice);
  }
  return result;
}

function toToolChoice(
  choice: NonNullable<ChatCompletionRequest['tool_choice']>,
): AnthropicToolChoice {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'required') return { type: 'any' };
  if (choice === 'none') return { type: 'none' };
  return { type: 'tool', name: choice.function.name };
}

/**
 * Anthropic requires strictly alternating user/assistant turns and carries
 * tool results inside user turns, so OpenAI `tool` messages become
 * `tool_result` blocks and consecutive same-role messages are merged.
 */
function toAnthropicMessages(messages: ChatMessage[]): AnthropicMessage[] {
  const result: AnthropicMessage[] = [];
  const append = (role: AnthropicMessage['role'], blocks: AnthropicContentBlock[]): void => {
    if (blocks.length === 0) return;
    const last = result.at(-1);
    if (last?.role === role) last.content.push(...blocks);
    else result.push({ role, content: blocks });
  };

  for (const message of messages) {
    switch (message.role) {
      case 'system':
      case 'developer':
        break;
      case 'user':
        append(
          'user',
          typeof message.content === 'string'
            ? [{ type: 'text', text: message.content }]
            : message.content.map(toAnthropicPart),
        );
        break;
      case 'assistant': {
        const blocks: AnthropicContentBlock[] = [];
        const text = message.content ? contentToText(message.content) : '';
        if (text) blocks.push({ type: 'text', text });
        for (const call of message.tool_calls ?? []) {
          blocks.push({
            type: 'tool_use',
            id: call.id,
            name: call.function.name,
            input: parseToolArguments(call),
          });
        }
        append('assistant', blocks);
        break;
      }
      case 'tool':
        append('user', [
          {
            type: 'tool_result',
            tool_use_id: message.tool_call_id,
            content: contentToText(message.content),
          },
        ]);
        break;
    }
  }
  return result;
}

const DATA_URL = /^data:(image\/(?:png|jpeg|gif|webp));base64,(.+)$/;

function toAnthropicPart(part: ContentPart): AnthropicContentBlock {
  if (part.type === 'text') return { type: 'text', text: part.text };
  const url = part.image_url.url;
  const match = DATA_URL.exec(url);
  if (match?.[1] && match[2]) {
    return { type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } };
  }
  if (/^https?:\/\//.test(url)) return { type: 'image', source: { type: 'url', url } };
  throw ApiError.invalidRequest(
    'image_url must be an http(s) URL or a base64 data URL (png, jpeg, gif, webp)',
    'messages.content.image_url.url',
  );
}

function parseToolArguments(call: ToolCall): unknown {
  try {
    return JSON.parse(call.function.arguments || '{}');
  } catch {
    throw ApiError.invalidRequest(
      `tool_calls[${call.id}].function.arguments is not valid JSON`,
      'messages.tool_calls.function.arguments',
    );
  }
}

export function mapStopReason(reason: AnthropicStopReason): FinishReason | null {
  switch (reason) {
    case 'end_turn':
    case 'stop_sequence':
    case 'pause_turn':
      return 'stop';
    case 'max_tokens':
      return 'length';
    case 'tool_use':
      return 'tool_calls';
    case 'refusal':
      return 'content_filter';
    default:
      return null;
  }
}

/** Prompt caching splits input tokens across three counters; all are billed input. */
function toUsage(usage: AnthropicUsage): Usage {
  const prompt =
    (usage.input_tokens ?? 0) +
    (usage.cache_creation_input_tokens ?? 0) +
    (usage.cache_read_input_tokens ?? 0);
  const completion = usage.output_tokens ?? 0;
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
  };
}

export function fromAnthropicResponse(
  response: AnthropicResponse,
  created: number,
): ChatCompletion {
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const block of response.content) {
    if (block.type === 'text' && 'text' in block) text += block.text;
    if (block.type === 'tool_use' && 'id' in block) {
      toolCalls.push({
        id: block.id,
        type: 'function',
        function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
      });
    }
  }
  return {
    id: response.id,
    object: 'chat.completion',
    created,
    model: response.model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text || (toolCalls.length > 0 ? null : ''),
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: mapStopReason(response.stop_reason),
      },
    ],
    usage: toUsage(response.usage),
  };
}

/**
 * Stateful translator from Anthropic stream events to OpenAI chunks.
 *
 * Anthropic numbers content blocks across text and tool_use together, while
 * OpenAI numbers tool calls separately, so tool block indexes are remapped.
 * Usage arrives split across message_start (input) and message_delta
 * (output) and is emitted as a final usage-only chunk at message_stop.
 */
export class AnthropicStreamTranslator {
  private id = '';
  private model: string;
  private usage: AnthropicUsage = {};
  private done = false;
  private readonly toolIndexByBlock = new Map<number, number>();

  constructor(
    model: string,
    private readonly created: number,
  ) {
    this.model = model;
  }

  /** True once message_stop has been seen. */
  get finished(): boolean {
    return this.done;
  }

  translate(event: AnthropicStreamEvent): ChatCompletionChunk[] {
    switch (event.type) {
      case 'message_start':
        this.id = event.message.id;
        this.model = event.message.model;
        this.usage = { ...event.message.usage };
        return [this.chunk({ role: 'assistant', content: '' })];

      case 'content_block_start': {
        const block = event.content_block;
        if (block.type === 'tool_use' && 'id' in block) {
          const toolIndex = this.toolIndexByBlock.size;
          this.toolIndexByBlock.set(event.index, toolIndex);
          return [
            this.chunk({
              tool_calls: [
                {
                  index: toolIndex,
                  id: block.id,
                  type: 'function',
                  function: { name: block.name, arguments: '' },
                },
              ],
            }),
          ];
        }
        if (block.type === 'text' && 'text' in block && block.text) {
          return [this.chunk({ content: block.text })];
        }
        return [];
      }

      case 'content_block_delta': {
        const delta = event.delta;
        if (delta.type === 'text_delta' && 'text' in delta) {
          return [this.chunk({ content: delta.text })];
        }
        if (delta.type === 'input_json_delta' && 'partial_json' in delta) {
          const toolIndex = this.toolIndexByBlock.get(event.index);
          if (toolIndex === undefined || delta.partial_json === '') return [];
          return [
            this.chunk({
              tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }],
            }),
          ];
        }
        // thinking/signature deltas have no OpenAI equivalent.
        return [];
      }

      case 'message_delta':
        if (event.usage) this.usage = { ...this.usage, ...definedOnly(event.usage) };
        return [this.chunk({}, mapStopReason(event.delta.stop_reason))];

      case 'message_stop':
        this.done = true;
        return [
          {
            id: this.id,
            object: 'chat.completion.chunk',
            created: this.created,
            model: this.model,
            choices: [],
            usage: toUsage(this.usage),
          },
        ];

      case 'error':
        throw new UpstreamError(
          'anthropic',
          event.error.type === 'invalid_request_error' ? 'client' : 'server',
          `anthropic stream error (${event.error.type}): ${event.error.message}`,
        );

      default:
        return [];
    }
  }

  private chunk(
    delta: ChatCompletionChunk['choices'][number]['delta'],
    finishReason: FinishReason | null = null,
  ): ChatCompletionChunk {
    return {
      id: this.id,
      object: 'chat.completion.chunk',
      created: this.created,
      model: this.model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    };
  }
}

function definedOnly(usage: AnthropicUsage): AnthropicUsage {
  return Object.fromEntries(
    Object.entries(usage).filter(([, value]) => value !== undefined && value !== null),
  );
}
