# @aigw/sdk

A small, typed client for the gateway. It uses the platform `fetch`, so the same code runs in Node
and in browsers, with no runtime dependencies beyond `@aigw/shared` (the shared zod contracts). It
relies on `AbortSignal.any` (Node 20.3+, Chrome 116+, Firefox 124+, Safari 17.4+).

## Chat completions

```ts
import { GatewayClient, RateLimitError, StreamError } from '@aigw/sdk';

const gateway = new GatewayClient({
  baseUrl: 'http://localhost:3000',
  apiKey: process.env.GATEWAY_API_KEY!, // gw_<prefix>_<secret>
  maxRetries: 2, // 408/409/429/5xx and connection errors; honors Retry-After
  timeoutMs: 60_000, // time to response headers (and full body when not streaming)
});

// Non-streaming
const completion = await gateway.chat.completions.create({
  model: 'smart', // an alias from the gateway's route config, not a provider model ID
  messages: [{ role: 'user', content: 'Summarize RFC 9110 in one sentence.' }],
});
console.log(completion.choices[0]?.message.content);

// Streaming: an async iterator over parsed chunks
const controller = new AbortController();
try {
  for await (const chunk of gateway.chat.completions.stream(
    { model: 'fast', messages: [{ role: 'user', content: 'Write a haiku about queues.' }] },
    { signal: controller.signal }, // aborting also cancels the upstream provider call
  )) {
    process.stdout.write(chunk.choices[0]?.delta.content ?? '');
  }
} catch (error) {
  if (error instanceof RateLimitError) console.error('Slow down; retry after', error.retryAfter);
  else if (error instanceof StreamError) console.error('Stream interrupted:', error.message);
  else throw error;
}

const models = await gateway.models.list();
```

## Admin API

`GatewayAdminClient` wraps the admin endpoints used by the dashboard (usage analytics, routes, key
management). Key creation and updates are never retried automatically because they are not
idempotent.

```ts
import { GatewayAdminClient } from '@aigw/sdk';

const admin = new GatewayAdminClient({ baseUrl: 'http://localhost:3000', adminToken });
const { secret } = await admin.createKey({ name: 'ci', project: 'demo', monthlyBudgetUsd: 20 });
const page = await admin.listRequests({ status: 'error', limit: 20 });
```

## Errors

| Class                                                                              | When                                                        |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError` | 400/422, 401, 403, 404                                      |
| `RateLimitError`                                                                   | 429 (gateway rate limit, exhausted budget, or upstream 429) |
| `InternalServerError`                                                              | 5xx after retries                                           |
| `APIConnectionError`, `APITimeoutError`                                            | no response / deadline exceeded                             |
| `StreamError`                                                                      | the gateway reported an error after the stream started      |

All HTTP errors extend `APIError` and expose `status`, `type`, `code`, `param` and the gateway's
`requestId` (from `x-request-id`) for support correlation.

## Streaming details

The SSE parser (`readSseStream` in `@aigw/shared`) follows the WHATWG event-stream rules: it handles
events split across network chunks, CRLF pairs split across chunks and multi-byte UTF-8 boundaries,
and it cancels the underlying body when the consumer stops iterating early. It uses the stream reader
API rather than async iteration of `ReadableStream`, which Safari does not implement.
