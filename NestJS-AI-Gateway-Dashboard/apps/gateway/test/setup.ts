import { Logger } from '@nestjs/common';

// Expected failures (scripted upstream errors, simulated DB outages) log
// warnings by design; keep test output focused on assertions.
Logger.overrideLogger(false);
