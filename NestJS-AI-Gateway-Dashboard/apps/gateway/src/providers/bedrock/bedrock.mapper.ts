import type {
  ContentBlock,
  ConverseRequest,
  ConverseResponse,
  ConverseStreamOutput,
  ImageFormat,
  Message,
  StopReason,
  TokenUsage,
  Tool,
  ToolChoice,
  ToolUseBlock,
} from '@aws-sdk/client-bedrock-runtime';
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

type JsonDocument = NonNullable<ToolUseBlock['input']>;

/** Converse request fields the mapper fills in; the provider adds modelId. */
export type BedrockConverseInput = Omit<ConverseRequest, 'modelId'>;

/**
 * Translates an OpenAI-format request into a Bedrock Converse request. The
 * Converse API is model-agnostic, so this one mapping covers every Bedrock
 * model that supports Converse.
 */
export function toConverseInput(
  request: ChatCompletionRequest,
  defaultMaxTokens: number,
): BedrockConverseInput {
  const system = request.messages.flatMap((m) =>
    m.role === 'system' || m.role === 'developer' ? [{ text: contentToText(m.content) }] : [],
  );

  const input: BedrockConverseInput = {
    messages: toConverseMessages(request.messages),
    inferenceConfig: {
      maxTokens: request.max_completion_tokens ?? request.max_tokens ?? defaultMaxTokens,
      ...(request.temperature !== undefined
        ? { temperature: Math.min(request.temperature, 1) }
        : {}),
      ...(request.top_p !== undefined ? { topP: request.top_p } : {}),
      ...(request.stop !== undefined
        ? { stopSequences: typeof request.stop === 'string' ? [request.stop] : request.stop }
        : {}),
    },
  };
  if (system.length > 0) input.system = system;

  // Converse has no "none" tool choice; omitting the tools is the closest
  // equivalent, and is only safe when the history contains no tool blocks.
  const historyUsesTools = request.messages.some(
    (m) => m.role === 'tool' || (m.role === 'assistant' && (m.tool_calls?.length ?? 0) > 0),
  );
  const dropTools = request.tool_choice === 'none' && !historyUsesTools;
  if (request.tools?.length && !dropTools) {
    const tools: Tool[] = request.tools.map((tool) => ({
      toolSpec: {
        name: tool.function.name,
        ...(tool.function.description ? { description: tool.function.description } : {}),
        inputSchema: {
          json: (tool.function.parameters ?? { type: 'object', properties: {} }) as JsonDocument,
        },
      },
    }));
    const toolChoice = toToolChoice(request.tool_choice);
    input.toolConfig = { tools, ...(toolChoice ? { toolChoice } : {}) };
  }
  return input;
}

function toToolChoice(choice: ChatCompletionRequest['tool_choice']): ToolChoice | undefined {
  if (choice === 'required') return { any: {} };
  if (choice === 'auto') return { auto: {} };
  if (typeof choice === 'object') return { tool: { name: choice.function.name } };
  return undefined;
}

function toConverseMessages(messages: ChatMessage[]): Message[] {
  const result: Message[] = [];
  const append = (role: 'user' | 'assistant', blocks: ContentBlock[]): void => {
    if (blocks.length === 0) return;
    const last = result.at(-1);
    if (last?.role === role) last.content?.push(...blocks);
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
            ? [{ text: message.content }]
            : message.content.map(toConversePart),
        );
        break;
      case 'assistant': {
        const blocks: ContentBlock[] = [];
        const text = message.content ? contentToText(message.content) : '';
        if (text) blocks.push({ text });
        for (const call of message.tool_calls ?? []) {
          blocks.push({
            toolUse: { toolUseId: call.id, name: call.function.name, input: parseArguments(call) },
          });
        }
        append('assistant', blocks);
        break;
      }
      case 'tool':
        append('user', [
          {
            toolResult: {
              toolUseId: message.tool_call_id,
              content: [{ text: contentToText(message.content) }],
            },
          },
        ]);
        break;
    }
  }
  return result;
}

const DATA_URL = /^data:image\/(png|jpeg|gif|webp);base64,(.+)$/;

function toConversePart(part: ContentPart): ContentBlock {
  if (part.type === 'text') return { text: part.text };
  // Converse takes raw bytes (or S3); it cannot fetch arbitrary URLs.
  const match = DATA_URL.exec(part.image_url.url);
  if (!match?.[1] || !match[2]) {
    throw ApiError.invalidRequest(
      'Bedrock routes accept images only as base64 data URLs (png, jpeg, gif, webp)',
      'messages.content.image_url.url',
    );
  }
  return {
    image: {
      format: match[1] as ImageFormat,
      source: { bytes: Buffer.from(match[2], 'base64') },
    },
  };
}

function parseArguments(call: ToolCall): JsonDocument {
  try {
    return JSON.parse(call.function.arguments || '{}') as JsonDocument;
  } catch {
    throw ApiError.invalidRequest(
      `tool_calls[${call.id}].function.arguments is not valid JSON`,
      'messages.tool_calls.function.arguments',
    );
  }
}

export function mapStopReason(reason: StopReason | undefined): FinishReason | null {
  switch (reason) {
    case 'end_turn':
    case 'stop_sequence':
      return 'stop';
    case 'max_tokens':
    case 'model_context_window_exceeded':
      return 'length';
    case 'tool_use':
      return 'tool_calls';
    case 'content_filtered':
    case 'guardrail_intervened':
      return 'content_filter';
    default:
      return reason === undefined ? null : 'stop';
  }
}

function toUsage(usage: TokenUsage | undefined): Usage {
  const prompt =
    (usage?.inputTokens ?? 0) +
    (usage?.cacheReadInputTokens ?? 0) +
    (usage?.cacheWriteInputTokens ?? 0);
  const completion = usage?.outputTokens ?? 0;
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
  };
}

export function fromConverseResponse(
  response: ConverseResponse,
  id: string,
  model: string,
  created: number,
): ChatCompletion {
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const block of response.output?.message?.content ?? []) {
    if (block.text !== undefined) text += block.text;
    if (block.toolUse) {
      toolCalls.push({
        id: block.toolUse.toolUseId ?? '',
        type: 'function',
        function: {
          name: block.toolUse.name ?? '',
          arguments: JSON.stringify(block.toolUse.input ?? {}),
        },
      });
    }
  }
  return {
    id,
    object: 'chat.completion',
    created,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text || (toolCalls.length > 0 ? null : ''),
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: mapStopReason(response.stopReason),
      },
    ],
    usage: toUsage(response.usage),
  };
}

/**
 * Translates ConverseStream events into OpenAI chunks. Bedrock reports usage
 * in a trailing `metadata` event, after `messageStop`, so the usage chunk is
 * emitted from there.
 */
export class ConverseStreamTranslator {
  private readonly toolIndexByBlock = new Map<number, number>();
  private done = false;

  constructor(
    private readonly id: string,
    private readonly model: string,
    private readonly created: number,
  ) {}

  get finished(): boolean {
    return this.done;
  }

  translate(event: ConverseStreamOutput): ChatCompletionChunk[] {
    if (event.messageStart) return [this.chunk({ role: 'assistant', content: '' })];

    if (event.contentBlockStart) {
      const toolUse = event.contentBlockStart.start?.toolUse;
      if (!toolUse) return [];
      const toolIndex = this.toolIndexByBlock.size;
      this.toolIndexByBlock.set(event.contentBlockStart.contentBlockIndex ?? 0, toolIndex);
      return [
        this.chunk({
          tool_calls: [
            {
              index: toolIndex,
              id: toolUse.toolUseId ?? '',
              type: 'function',
              function: { name: toolUse.name ?? '', arguments: '' },
            },
          ],
        }),
      ];
    }

    if (event.contentBlockDelta) {
      const delta = event.contentBlockDelta.delta;
      if (delta?.text !== undefined) return [this.chunk({ content: delta.text })];
      if (delta?.toolUse?.input) {
        const toolIndex = this.toolIndexByBlock.get(event.contentBlockDelta.contentBlockIndex ?? 0);
        if (toolIndex === undefined) return [];
        return [
          this.chunk({
            tool_calls: [{ index: toolIndex, function: { arguments: delta.toolUse.input } }],
          }),
        ];
      }
      return [];
    }

    if (event.messageStop) {
      return [this.chunk({}, mapStopReason(event.messageStop.stopReason))];
    }

    if (event.metadata) {
      this.done = true;
      return [
        {
          id: this.id,
          object: 'chat.completion.chunk',
          created: this.created,
          model: this.model,
          choices: [],
          usage: toUsage(event.metadata.usage),
        },
      ];
    }

    const failure =
      event.throttlingException ??
      event.serviceUnavailableException ??
      event.internalServerException ??
      event.modelStreamErrorException ??
      event.validationException;
    if (failure) {
      const kind = event.throttlingException
        ? 'rate_limit'
        : event.validationException
          ? 'client'
          : 'server';
      throw new UpstreamError('bedrock', kind, `bedrock stream error: ${failure.message}`);
    }
    return [];
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
