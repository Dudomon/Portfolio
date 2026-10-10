import { Module } from '@nestjs/common';
import { ProvidersModule } from '../providers/providers.module';
import { CircuitBreakerRegistry } from './circuit-breaker.registry';
import { RouterService } from './router.service';

@Module({
  imports: [ProvidersModule],
  providers: [RouterService, CircuitBreakerRegistry],
  exports: [RouterService, CircuitBreakerRegistry],
})
export class RoutingModule {}
