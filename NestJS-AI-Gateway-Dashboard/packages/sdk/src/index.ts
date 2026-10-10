export { GatewayClient, type GatewayClientOptions, type CallOptions } from './client';
export { GatewayAdminClient, type GatewayAdminClientOptions } from './admin-client';
export {
  APIConnectionError,
  APIError,
  APITimeoutError,
  AuthenticationError,
  BadRequestError,
  GatewayError,
  InternalServerError,
  NotFoundError,
  PermissionDeniedError,
  RateLimitError,
  StreamError,
} from './errors';
export type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatMessage,
  ModelList,
} from '@aigw/shared';
