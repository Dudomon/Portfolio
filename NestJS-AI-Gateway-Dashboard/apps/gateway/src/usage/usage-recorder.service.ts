import {
  Inject,
  Injectable,
  Logger,
  type BeforeApplicationShutdown,
  type OnModuleInit,
} from '@nestjs/common';
import type { RequestStatus } from '@aigw/shared';
import { ENV } from '../config/config.module';
import type { AppEnv } from '../config/env';
import { PrismaService } from '../infra/prisma.service';
import { MetricsService } from '../observability/metrics.service';

export interface UsageEntry {
  requestId: string;
  apiKeyId: string;
  projectId: string;
  alias: string;
  provider: string | null;
  model: string | null;
  status: RequestStatus;
  errorCode: string | null;
  promptTokens: number;
  completionTokens: number;
  costUsd: number;
  latencyMs: number;
  attempts: number;
  cached: boolean;
  streamed: boolean;
  createdAt: Date;
}

/**
 * Write-behind buffer for usage records.
 *
 * Requests enqueue synchronously and never wait on the database; records are
 * flushed with one multi-row insert when the batch fills or on an interval.
 * On database failure the batch is requeued, and the buffer is bounded: past
 * USAGE_BUFFER_MAX the oldest records are dropped (and counted) so an outage
 * degrades reporting instead of exhausting memory.
 *
 * Trade-off: a crash loses at most one interval of records. Budget counters
 * in Redis are updated immediately, so enforcement does not depend on this.
 */
@Injectable()
export class UsageRecorder implements OnModuleInit, BeforeApplicationShutdown {
  private readonly logger = new Logger(UsageRecorder.name);
  private buffer: UsageEntry[] = [];
  private timer: NodeJS.Timeout | undefined;
  private inFlight: Promise<void> | undefined;

  constructor(
    private readonly prisma: PrismaService,
    private readonly metrics: MetricsService,
    @Inject(ENV) private readonly env: AppEnv,
  ) {}

  onModuleInit(): void {
    this.timer = setInterval(() => void this.flush(), this.env.USAGE_FLUSH_INTERVAL_MS);
    // Do not keep the process alive just for the flush timer.
    this.timer.unref();
  }

  async beforeApplicationShutdown(): Promise<void> {
    clearInterval(this.timer);
    // Drain until empty, stopping if a flush makes no progress (DB down).
    let previous = Number.POSITIVE_INFINITY;
    while (this.buffer.length > 0 && this.buffer.length < previous) {
      previous = this.buffer.length;
      await this.flush();
    }
    if (this.buffer.length > 0) {
      this.logger.error(`Shutting down with ${this.buffer.length} unwritten usage records`);
    }
  }

  record(entry: UsageEntry): void {
    this.buffer.push(entry);
    const overflow = this.buffer.length - this.env.USAGE_BUFFER_MAX;
    if (overflow > 0) {
      this.buffer.splice(0, overflow);
      this.metrics.usageDropped.inc(overflow);
    }
    this.metrics.usageBufferSize.set(this.buffer.length);
    if (this.buffer.length >= this.env.USAGE_FLUSH_BATCH_SIZE) void this.flush();
  }

  /** Writes one batch. Concurrent callers share the in-flight flush. */
  flush(): Promise<void> {
    this.inFlight ??= this.writeBatch().finally(() => {
      this.inFlight = undefined;
    });
    return this.inFlight;
  }

  private async writeBatch(): Promise<void> {
    if (this.buffer.length === 0) return;
    const batch = this.buffer.splice(0, this.env.USAGE_FLUSH_BATCH_SIZE);
    try {
      await this.prisma.usageRecord.createMany({ data: batch });
    } catch (error) {
      this.logger.error(
        `Usage flush failed (${batch.length} records requeued): ${(error as Error).message}`,
      );
      this.buffer.unshift(...batch);
      const overflow = this.buffer.length - this.env.USAGE_BUFFER_MAX;
      if (overflow > 0) {
        this.buffer.splice(0, overflow);
        this.metrics.usageDropped.inc(overflow);
      }
    } finally {
      this.metrics.usageBufferSize.set(this.buffer.length);
    }
  }
}
