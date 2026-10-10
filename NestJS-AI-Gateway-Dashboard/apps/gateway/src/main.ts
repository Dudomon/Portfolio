// Order matters: environment first, then tracing, then everything it patches.
import './load-env';
import './tracing';
import 'reflect-metadata';
import { NestFactory } from '@nestjs/core';
import type { NestExpressApplication } from '@nestjs/platform-express';
import { Logger } from 'nestjs-pino';
import { AppModule } from './app.module';
import { ENV } from './config/config.module';
import type { AppEnv } from './config/env';

async function bootstrap(): Promise<void> {
  const app = await NestFactory.create<NestExpressApplication>(AppModule, {
    bufferLogs: true,
    bodyParser: false,
  });
  app.useLogger(app.get(Logger));

  const env = app.get<AppEnv>(ENV);
  // Chat requests with base64 images easily exceed Express's 100kb default.
  app.useBodyParser('json', { limit: '10mb' });
  app.disable('x-powered-by');
  if (env.CORS_ORIGINS) {
    app.enableCors({ origin: env.CORS_ORIGINS.split(',').map((origin) => origin.trim()) });
  }
  app.enableShutdownHooks();

  await app.listen(env.PORT);
}

void bootstrap();
