import { describe, expect, it, vi } from 'vitest';
import type { AppEnv } from '../src/config/env';
import type { PrismaService } from '../src/infra/prisma.service';
import { MetricsService } from '../src/observability/metrics.service';
import { UsageRecorder, type UsageEntry } from '../src/usage/usage-recorder.service';

function entry(requestId: string): UsageEntry {
  return {
    requestId,
    apiKeyId: 'k',
    projectId: 'p',
    alias: 'fast',
    provider: 'openai',
    model: 'm',
    status: 'success',
    errorCode: null,
    promptTokens: 1,
    completionTokens: 1,
    costUsd: 0.001,
    latencyMs: 10,
    attempts: 1,
    cached: false,
    streamed: false,
    createdAt: new Date(0),
  };
}

function setup(
  createMany: (args: { data: UsageEntry[] }) => Promise<unknown>,
  overrides: Partial<AppEnv> = {},
) {
  const prisma = { usageRecord: { createMany: vi.fn(createMany) } } as unknown as PrismaService;
  const env = {
    USAGE_FLUSH_INTERVAL_MS: 60_000,
    USAGE_FLUSH_BATCH_SIZE: 3,
    USAGE_BUFFER_MAX: 5,
    ...overrides,
  } as AppEnv;
  const metrics = new MetricsService();
  return { recorder: new UsageRecorder(prisma, metrics, env), prisma, metrics };
}

describe('UsageRecorder', () => {
  it('flushes automatically when a batch fills, without blocking record()', async () => {
    const written: string[] = [];
    const { recorder } = setup(({ data }) => {
      written.push(...data.map((d) => d.requestId));
      return Promise.resolve();
    });

    recorder.record(entry('a'));
    recorder.record(entry('b'));
    expect(written).toEqual([]);
    recorder.record(entry('c'));
    await recorder.flush();

    expect(written).toEqual(['a', 'b', 'c']);
  });

  it('requeues a failed batch in order and retries it later', async () => {
    const createMany = vi
      .fn<(args: { data: UsageEntry[] }) => Promise<unknown>>()
      .mockRejectedValueOnce(new Error('db down'))
      .mockResolvedValue({ count: 2 });
    const { recorder, prisma } = setup(createMany);

    recorder.record(entry('a'));
    recorder.record(entry('b'));
    await recorder.flush();
    await recorder.flush();

    const calls = vi.mocked(prisma.usageRecord.createMany).mock.calls;
    expect(calls).toHaveLength(2);
    expect(calls[1]?.[0]).toEqual({ data: [entry('a'), entry('b')] });
  });

  it('drops the oldest records past the buffer bound and counts them', async () => {
    const { recorder, metrics } = setup(() => Promise.reject(new Error('db down')), {
      USAGE_FLUSH_BATCH_SIZE: 100,
    });

    for (const id of ['1', '2', '3', '4', '5', '6', '7']) recorder.record(entry(id));

    expect((await metrics.usageDropped.get()).values[0]?.value).toBe(2);
    expect((await metrics.usageBufferSize.get()).values[0]?.value).toBe(5);
  });

  it('drains the buffer on shutdown and stops if the database makes no progress', async () => {
    const { recorder, prisma } = setup(() => Promise.reject(new Error('db down')));
    recorder.record(entry('a'));

    await recorder.beforeApplicationShutdown();

    expect(vi.mocked(prisma.usageRecord.createMany)).toHaveBeenCalledTimes(1);
  });
});
