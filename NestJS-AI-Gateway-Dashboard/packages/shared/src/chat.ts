import { z } from 'zod';

/**
 * OpenAI-compatible chat completion wire format.
 *
 * This is the gateway's canonical representation: clients speak it, and every
 * provider adapter translates to and from it. Only the subset of fields the
 * gateway can honor across all providers is accepted; unknown fields are
 * stripped rather than rejected so stock OpenAI clients keep working.
 */

const textPartSchema = z.object({
  type: z.literal('text'),
  text: z.string(),
});

const imagePartSchema = z.object({
  type: z.literal('image_url'),
  image_url: z.object({
    url: z.string().min(1),
    detail: z.enum(['auto', 'low', 'high']).optional(),
  }),
});

export const contentPartSchema = z.discriminatedUnion('type', [textPartSchema, imagePartSchema]);
export type ContentPart = z.infer<typeof contentPartSchema>;

export const toolCallSchema = z.object({
  id: z.string().min(1),
  type: z.literal('function'),
  function: z.object({
    name: z.string().min(1),
    // Arguments stay a JSON-encoded string, exactly as OpenAI returns them.
    arguments: z.string(),
  }),
});
export type ToolCall = z.infer<typeof toolCallSchema>;

const systemMessageSchema = z.object({
  role: z.enum(['system', 'developer']),
  content: z.union([z.string(), z.array(textPartSchema)]),
  name: z.string().optional(),
});

const userMessageSchema = z.object({
  role: z.literal('user'),
  content: z.union([z.string(), z.array(contentPartSchema).min(1)]),
  name: z.string().optional(),
});

const assistantMessageSchema = z.object({
  role: z.literal('assistant'),
  content: z
    .union([z.string(), z.array(textPartSchema)])
    .nullable()
    .optional(),
  tool_calls: z.array(toolCallSchema).optional(),
  name: z.string().optional(),
});

const toolMessageSchema = z.object({
  role: z.literal('tool'),
  tool_call_id: z.string().min(1),
  content: z.union([z.string(), z.array(textPartSchema)]),
});

export const chatMessageSchema = z.discriminatedUnion('role', [
  systemMessageSchema,
  userMessageSchema,
  assistantMessageSchema,
  toolMessageSchema,
]);
export type ChatMessage = z.infer<typeof chatMessageSchema>;
export type AssistantMessage = z.infer<typeof assistantMessageSchema>;

export const functionToolSchema = z.object({
  type: z.literal('function'),
  function: z.object({
    name: z.string().regex(/^[a-zA-Z0-9_-]{1,64}$/, 'Tool names must match ^[a-zA-Z0-9_-]{1,64}$'),
    description: z.string().optional(),
    parameters: z.record(z.string(), z.unknown()).optional(),
  }),
});
export type FunctionTool = z.infer<typeof functionToolSchema>;

export const toolChoiceSchema = z.union([
  z.enum(['none', 'auto', 'required']),
  z.object({
    type: z.literal('function'),
    function: z.object({ name: z.string().min(1) }),
  }),
]);
export type ToolChoice = z.infer<typeof toolChoiceSchema>;

export const chatCompletionRequestSchema = z
  .object({
    model: z.string().min(1),
    messages: z.array(chatMessageSchema).min(1),
    temperature: z.number().min(0).max(2).optional(),
    top_p: z.number().gt(0).max(1).optional(),
    max_tokens: z.number().int().positive().optional(),
    max_completion_tokens: z.number().int().positive().optional(),
    stop: z.union([z.string(), z.array(z.string()).max(4)]).optional(),
    stream: z.boolean().optional(),
    stream_options: z.object({ include_usage: z.boolean().optional() }).optional(),
    tools: z.array(functionToolSchema).optional(),
    tool_choice: toolChoiceSchema.optional(),
    // Multiple choices cannot be emulated on providers that lack them, so the
    // gateway accepts `n` only for compatibility and requires it to be 1.
    n: z.literal(1).optional(),
    user: z.string().max(256).optional(),
  })
  .refine((req) => req.tool_choice === undefined || req.tool_choice === 'none' || req.tools, {
    message: 'tool_choice requires tools to be provided',
    path: ['tool_choice'],
  });
export type ChatCompletionRequest = z.infer<typeof chatCompletionRequestSchema>;

export const finishReasonSchema = z.enum(['stop', 'length', 'tool_calls', 'content_filter']);
export type FinishReason = z.infer<typeof finishReasonSchema>;

export const usageSchema = z.object({
  prompt_tokens: z.number().int().nonnegative(),
  completion_tokens: z.number().int().nonnegative(),
  total_tokens: z.number().int().nonnegative(),
});
export type Usage = z.infer<typeof usageSchema>;

export const chatCompletionSchema = z.object({
  id: z.string(),
  object: z.literal('chat.completion'),
  created: z.number().int(),
  model: z.string(),
  choices: z.array(
    z.object({
      index: z.number().int(),
      message: z.object({
        role: z.literal('assistant'),
        content: z.string().nullable(),
        tool_calls: z.array(toolCallSchema).optional(),
      }),
      finish_reason: finishReasonSchema.nullable(),
    }),
  ),
  usage: usageSchema.optional(),
});
export type ChatCompletion = z.infer<typeof chatCompletionSchema>;

export const toolCallDeltaSchema = z.object({
  index: z.number().int(),
  id: z.string().optional(),
  type: z.literal('function').optional(),
  function: z
    .object({
      name: z.string().optional(),
      arguments: z.string().optional(),
    })
    .optional(),
});
export type ToolCallDelta = z.infer<typeof toolCallDeltaSchema>;

export const chatCompletionChunkSchema = z.object({
  id: z.string(),
  object: z.literal('chat.completion.chunk'),
  created: z.number().int(),
  model: z.string(),
  choices: z.array(
    z.object({
      index: z.number().int(),
      delta: z.object({
        role: z.literal('assistant').optional(),
        content: z.string().nullable().optional(),
        tool_calls: z.array(toolCallDeltaSchema).optional(),
      }),
      finish_reason: finishReasonSchema.nullable(),
    }),
  ),
  usage: usageSchema.nullable().optional(),
});
export type ChatCompletionChunk = z.infer<typeof chatCompletionChunkSchema>;

export const modelSchema = z.object({
  id: z.string(),
  object: z.literal('model'),
  created: z.number().int(),
  owned_by: z.string(),
});
export type Model = z.infer<typeof modelSchema>;

export const modelListSchema = z.object({
  object: z.literal('list'),
  data: z.array(modelSchema),
});
export type ModelList = z.infer<typeof modelListSchema>;

/** Flattens string-or-parts content into plain text. */
export function contentToText(
  content: string | ReadonlyArray<{ type: string; text?: string }>,
): string {
  if (typeof content === 'string') return content;
  return content
    .map((part) => (part.type === 'text' && part.text !== undefined ? part.text : ''))
    .join('');
}
