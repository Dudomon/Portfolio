/**
 * The subset of the Anthropic Messages API the adapter uses
 * (https://docs.anthropic.com/en/api/messages). Declared locally rather than
 * pulling in the SDK because the adapter needs only these wire shapes.
 */

export type AnthropicImageSource =
  { type: 'base64'; media_type: string; data: string } | { type: 'url'; url: string };

export type AnthropicContentBlock =
  | { type: 'text'; text: string }
  | { type: 'image'; source: AnthropicImageSource }
  | { type: 'tool_use'; id: string; name: string; input: unknown }
  | { type: 'tool_result'; tool_use_id: string; content: string };

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: AnthropicContentBlock[];
}

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}

export type AnthropicToolChoice =
  { type: 'auto' } | { type: 'any' } | { type: 'none' } | { type: 'tool'; name: string };

export interface AnthropicRequest {
  model: string;
  max_tokens: number;
  messages: AnthropicMessage[];
  system?: string;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  tools?: AnthropicTool[];
  tool_choice?: AnthropicToolChoice;
  metadata?: { user_id: string };
  stream?: boolean;
}

export interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
}

export type AnthropicStopReason =
  'end_turn' | 'max_tokens' | 'stop_sequence' | 'tool_use' | 'pause_turn' | 'refusal' | null;

/** Response blocks may include types the gateway ignores (e.g. thinking). */
export type AnthropicResponseBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: unknown }
  | { type: string };

export interface AnthropicResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  model: string;
  content: AnthropicResponseBlock[];
  stop_reason: AnthropicStopReason;
  usage: AnthropicUsage;
}

export type AnthropicStreamEvent =
  | { type: 'message_start'; message: { id: string; model: string; usage: AnthropicUsage } }
  | {
      type: 'content_block_start';
      index: number;
      content_block:
        | { type: 'text'; text: string }
        | { type: 'tool_use'; id: string; name: string }
        | { type: string };
    }
  | {
      type: 'content_block_delta';
      index: number;
      delta:
        | { type: 'text_delta'; text: string }
        | { type: 'input_json_delta'; partial_json: string }
        | { type: string };
    }
  | { type: 'content_block_stop'; index: number }
  | { type: 'message_delta'; delta: { stop_reason: AnthropicStopReason }; usage?: AnthropicUsage }
  | { type: 'message_stop' }
  | { type: 'ping' }
  | { type: 'error'; error: { type: string; message: string } };
