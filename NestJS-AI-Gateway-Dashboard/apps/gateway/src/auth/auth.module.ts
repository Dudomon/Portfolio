import { Module } from '@nestjs/common';
import { AdminGuard } from './admin.guard';
import { ApiKeyGuard } from './api-key.guard';
import { ApiKeyService } from './api-key.service';

@Module({
  providers: [ApiKeyService, ApiKeyGuard, AdminGuard],
  exports: [ApiKeyService, ApiKeyGuard, AdminGuard],
})
export class AuthModule {}
