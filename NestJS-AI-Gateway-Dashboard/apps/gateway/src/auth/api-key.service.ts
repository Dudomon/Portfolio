import { createHmac, randomBytes, timingSafeEqual } from 'node:crypto';
import { Inject, Injectable } from '@nestjs/common';
import { ENV } from '../config/config.module';
import type { AppEnv } from '../config/env';
import { PrismaService } from '../infra/prisma.service';

/** The authenticated caller, attached to the request by {@link ApiKeyGuard}. */
export interface AuthenticatedKey {
  id: string;
  name: string;
  prefix: string;
  projectId: string;
  rateLimitPerMinute: number;
  monthlyBudgetUsd: number | null;
}

interface CacheEntry {
  key: AuthenticatedKey;
  hash: string;
  revoked: boolean;
  expiresAt: number;
}

const KEY_PATTERN = /^gw_([0-9a-f]{12})_([A-Za-z0-9_-]{43})$/;
const MAX_CACHE_ENTRIES = 10_000;

/**
 * Issues and verifies API keys of the form `gw_<prefix>_<secret>`.
 *
 * - The prefix is a non-secret lookup handle with a unique index, so
 *   verification is one indexed read rather than a scan over hashes.
 * - Only HMAC-SHA256(pepper, key) is stored. The pepper lives outside the
 *   database, so a leaked table alone cannot be used to verify guesses.
 *   A fast hash is appropriate because keys carry 256 bits of entropy; slow
 *   KDFs exist to protect low-entropy passwords.
 * - Lookups are cached briefly. Revocation is therefore effective within
 *   API_KEY_CACHE_TTL_MS on other replicas (immediately on the one that
 *   handled the revoke).
 */
@Injectable()
export class ApiKeyService {
  private readonly cache = new Map<string, CacheEntry>();

  constructor(
    private readonly prisma: PrismaService,
    @Inject(ENV) private readonly env: AppEnv,
  ) {}

  generate(): { plaintext: string; prefix: string; hash: string } {
    const prefix = randomBytes(6).toString('hex');
    const secret = randomBytes(32).toString('base64url');
    const plaintext = `gw_${prefix}_${secret}`;
    return { plaintext, prefix, hash: this.hash(plaintext) };
  }

  async authenticate(presented: string): Promise<AuthenticatedKey | null> {
    const match = KEY_PATTERN.exec(presented);
    const prefix = match?.[1];
    if (!prefix) return null;

    const entry = await this.lookup(prefix);
    if (!entry || entry.revoked) return null;

    const presentedHash = Buffer.from(this.hash(presented), 'hex');
    const storedHash = Buffer.from(entry.hash, 'hex');
    if (presentedHash.length !== storedHash.length || !timingSafeEqual(presentedHash, storedHash)) {
      return null;
    }
    return entry.key;
  }

  invalidate(prefix: string): void {
    this.cache.delete(prefix);
  }

  private hash(plaintext: string): string {
    return createHmac('sha256', this.env.API_KEY_PEPPER).update(plaintext).digest('hex');
  }

  private async lookup(prefix: string): Promise<CacheEntry | null> {
    const cached = this.cache.get(prefix);
    if (cached && cached.expiresAt > Date.now()) return cached;

    const row = await this.prisma.apiKey.findUnique({ where: { prefix } });
    if (!row) {
      this.cache.delete(prefix);
      return null;
    }

    const entry: CacheEntry = {
      key: {
        id: row.id,
        name: row.name,
        prefix: row.prefix,
        projectId: row.projectId,
        rateLimitPerMinute: row.rateLimitPerMinute,
        monthlyBudgetUsd: row.monthlyBudgetUsd === null ? null : row.monthlyBudgetUsd.toNumber(),
      },
      hash: row.hash,
      revoked: row.revokedAt !== null,
      expiresAt: Date.now() + this.env.API_KEY_CACHE_TTL_MS,
    };
    // Map preserves insertion order, so deleting the first key evicts the oldest.
    if (this.cache.size >= MAX_CACHE_ENTRIES) {
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) this.cache.delete(oldest);
    }
    this.cache.set(prefix, entry);
    return entry;
  }
}
