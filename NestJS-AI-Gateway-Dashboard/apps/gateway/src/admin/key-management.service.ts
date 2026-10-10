import { Injectable } from '@nestjs/common';
import type { ApiKey, Project } from '@prisma/client';
import type {
  ApiKeyView,
  CreateApiKeyResponse,
  UpdateApiKeyRequest,
  createApiKeyRequestSchema,
} from '@aigw/shared';
import type { z } from 'zod';
import { ApiKeyService } from '../auth/api-key.service';
import { ApiError } from '../common/api-error';
import { PrismaService } from '../infra/prisma.service';
import { monthStart } from '../limits/budget.service';

type KeyWithProject = ApiKey & { project: Project };

@Injectable()
export class KeyManagementService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly apiKeys: ApiKeyService,
  ) {}

  async list(): Promise<ApiKeyView[]> {
    const [keys, spend] = await Promise.all([
      this.prisma.apiKey.findMany({
        include: { project: true },
        orderBy: [{ revokedAt: { sort: 'desc', nulls: 'first' } }, { createdAt: 'desc' }],
      }),
      this.prisma.usageRecord.groupBy({
        by: ['apiKeyId'],
        where: { createdAt: { gte: monthStart(new Date()) } },
        _sum: { costUsd: true },
      }),
    ]);
    const spendByKey = new Map(spend.map((s) => [s.apiKeyId, s._sum.costUsd?.toNumber() ?? 0]));
    return keys.map((key) => toView(key, spendByKey.get(key.id) ?? 0));
  }

  async create(input: z.output<typeof createApiKeyRequestSchema>): Promise<CreateApiKeyResponse> {
    const { plaintext, prefix, hash } = this.apiKeys.generate();
    const key = await this.prisma.apiKey.create({
      data: {
        name: input.name,
        prefix,
        hash,
        monthlyBudgetUsd: input.monthlyBudgetUsd ?? null,
        rateLimitPerMinute: input.rateLimitPerMinute,
        project: {
          connectOrCreate: { where: { name: input.project }, create: { name: input.project } },
        },
      },
      include: { project: true },
    });
    return { key: toView(key, 0), secret: plaintext };
  }

  async update(id: string, input: UpdateApiKeyRequest): Promise<ApiKeyView> {
    const key = await this.prisma.apiKey
      .update({
        where: { id },
        data: {
          ...(input.name !== undefined ? { name: input.name } : {}),
          ...(input.monthlyBudgetUsd !== undefined
            ? { monthlyBudgetUsd: input.monthlyBudgetUsd }
            : {}),
          ...(input.rateLimitPerMinute !== undefined
            ? { rateLimitPerMinute: input.rateLimitPerMinute }
            : {}),
        },
        include: { project: true },
      })
      .catch(rethrowNotFound(id));
    this.apiKeys.invalidate(key.prefix);
    return toView(key, await this.monthSpend(id));
  }

  /** Soft delete: usage history must keep pointing at the key. Idempotent. */
  async revoke(id: string): Promise<ApiKeyView> {
    const existing = await this.prisma.apiKey.findUnique({ where: { id } });
    if (!existing) throw ApiError.notFound(`API key ${id} not found`);
    const key = await this.prisma.apiKey.update({
      where: { id },
      data: { revokedAt: existing.revokedAt ?? new Date() },
      include: { project: true },
    });
    this.apiKeys.invalidate(key.prefix);
    return toView(key, await this.monthSpend(id));
  }

  private async monthSpend(id: string): Promise<number> {
    const result = await this.prisma.usageRecord.aggregate({
      where: { apiKeyId: id, createdAt: { gte: monthStart(new Date()) } },
      _sum: { costUsd: true },
    });
    return result._sum.costUsd?.toNumber() ?? 0;
  }
}

function toView(key: KeyWithProject, monthToDateSpendUsd: number): ApiKeyView {
  return {
    id: key.id,
    name: key.name,
    project: key.project.name,
    prefix: key.prefix,
    monthlyBudgetUsd: key.monthlyBudgetUsd?.toNumber() ?? null,
    rateLimitPerMinute: key.rateLimitPerMinute,
    monthToDateSpendUsd,
    createdAt: key.createdAt.toISOString(),
    revokedAt: key.revokedAt?.toISOString() ?? null,
  };
}

function rethrowNotFound(id: string) {
  return (error: unknown): never => {
    // P2025: "record to update not found".
    if (error instanceof Error && 'code' in error && error.code === 'P2025') {
      throw ApiError.notFound(`API key ${id} not found`);
    }
    throw error;
  };
}
