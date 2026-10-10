import { once } from 'node:events';
import { Body, Controller, Get, Inject, Post, Req, Res, UseGuards } from '@nestjs/common';
import {
  chatCompletionRequestSchema,
  errorBody,
  type ChatCompletionRequest,
  type GatewayConfig,
  type ModelList,
  type RouteTarget,
} from '@aigw/shared';
import type { Request, Response } from 'express';
import type { AuthenticatedKey } from '../auth/api-key.service';
import { ApiKeyGuard } from '../auth/api-key.guard';
import { CurrentKey } from '../auth/request-context';
import { ApiError } from '../common/api-error';
import { unixSeconds } from '../common/clock';
import { ZodValidationPipe } from '../common/zod-validation.pipe';
import { GATEWAY_CONFIG } from '../config/config.module';
import { BudgetGuard, RateLimitGuard } from '../limits/limits.guards';
import { RouterService } from '../routing/router.service';
import { ChatService } from './chat.service';

/** OpenAI-compatible surface: `/v1/chat/completions` and `/v1/models`. */
@Controller('v1')
@UseGuards(ApiKeyGuard, RateLimitGuard, BudgetGuard)
export class ChatController {
  private readonly startedAt = unixSeconds();

  constructor(
    private readonly chat: ChatService,
    private readonly router: RouterService,
    @Inject(GATEWAY_CONFIG) private readonly config: GatewayConfig,
  ) {}

  @Get('models')
  listModels(): ModelList {
    // Only aliases with at least one usable provider are advertised.
    const data = Object.entries(this.config.routes)
      .filter(([, route]) => route.targets.some((t) => this.router.isEnabled(t.provider)))
      .map(([alias]) => ({
        id: alias,
        object: 'model' as const,
        created: this.startedAt,
        owned_by: 'gateway',
      }));
    return { object: 'list', data };
  }

  /**
   * Uses the raw response because streaming needs manual control over
   * headers, flushing and backpressure; non-streaming replies go through the
   * same path for consistent gateway headers.
   */
  @Post('chat/completions')
  async createCompletion(
    @Body(new ZodValidationPipe(chatCompletionRequestSchema)) body: ChatCompletionRequest,
    @CurrentKey() key: AuthenticatedKey,
    @Req() req: Request,
    @Res() res: Response,
  ): Promise<void> {
    // Abort upstream work as soon as the client goes away.
    const abort = new AbortController();
    res.on('close', () => {
      if (!res.writableFinished) abort.abort(new Error('client disconnected'));
    });
    // genReqId (observability.module.ts) always assigns a string ID.
    const context = { signal: abort.signal, requestId: req.id as string };

    if (!body.stream) {
      const result = await this.chat.complete(body, key, context);
      setRoutingHeaders(res, result.target, result.attempts);
      res.setHeader('x-gateway-cache', result.cached ? 'hit' : 'miss');
      res.status(200).json(result.completion);
      return;
    }

    // Errors before this point propagate to the exception filter and become
    // a normal JSON error response; after headers are sent they go in-band.
    const stream = await this.chat.stream(body, key, context);
    setRoutingHeaders(res, stream.target, stream.attempts);
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      // Disables proxy buffering (nginx) so tokens reach the client promptly.
      'x-accel-buffering': 'no',
    });

    try {
      for await (const chunk of stream.chunks) {
        if (abort.signal.aborted) break;
        await writeEvent(res, JSON.stringify(chunk), abort.signal);
      }
      if (!abort.signal.aborted) await writeEvent(res, '[DONE]', abort.signal);
    } catch (error) {
      if (!abort.signal.aborted) {
        await writeEvent(res, JSON.stringify(inBandError(error)), abort.signal).catch(
          () => undefined,
        );
      }
    } finally {
      res.end();
    }
  }
}

function setRoutingHeaders(res: Response, target: RouteTarget | null, attempts: number): void {
  if (target) {
    res.setHeader('x-gateway-provider', target.provider);
    res.setHeader('x-gateway-model', target.model);
  }
  res.setHeader('x-gateway-attempts', String(attempts));
}

/**
 * Writes one SSE event. When the socket buffer is full, waits for it to
 * drain so a slow client applies backpressure to the upstream stream instead
 * of the gateway buffering unbounded output; the wait ends early on abort.
 */
async function writeEvent(res: Response, data: string, signal: AbortSignal): Promise<void> {
  if (!res.write(`data: ${data}\n\n`)) {
    await once(res, 'drain', { signal });
  }
}

function inBandError(error: unknown) {
  if (error instanceof ApiError) {
    return errorBody(error.type, error.message, { code: error.code });
  }
  return errorBody('upstream_error', 'The upstream provider failed mid-stream', {
    code: 'stream_interrupted',
  });
}
