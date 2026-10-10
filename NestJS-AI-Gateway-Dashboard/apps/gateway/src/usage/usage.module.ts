import { Module } from '@nestjs/common';
import { UsageRecorder } from './usage-recorder.service';

@Module({
  providers: [UsageRecorder],
  exports: [UsageRecorder],
})
export class UsageModule {}
