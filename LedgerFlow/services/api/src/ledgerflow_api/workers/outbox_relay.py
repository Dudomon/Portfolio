"""Outbox relay: moves committed outbox rows onto RabbitMQ (at-least-once).

Each iteration claims a batch with ``FOR UPDATE SKIP LOCKED``, so several relay replicas can
run side by side without publishing the same row concurrently. A row is marked published only
after the broker confirms it; a crash between confirm and commit republishes the row, which is
why consumers deduplicate on ``event_id``.

Run with ``python -m ledgerflow_api.workers.outbox_relay``.
"""

import asyncio
import contextlib
import json
import signal
from datetime import UTC, datetime

import aio_pika
import structlog
from aio_pika.abc import AbstractExchange
from sqlalchemy import select

from ledgerflow_api.config import Settings, get_settings
from ledgerflow_api.db.models import OutboxEvent
from ledgerflow_api.db.session import (
    SessionFactory,
    create_engine,
    create_session_factory,
    system_session,
)
from ledgerflow_api.observability import configure_logging

logger = structlog.get_logger(__name__)

_MAX_ERROR_LENGTH = 2000


def build_message(event: OutboxEvent) -> aio_pika.Message:
    trace_parent = event.payload.get("trace_parent")
    return aio_pika.Message(
        body=json.dumps(event.payload, separators=(",", ":")).encode(),
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        message_id=str(event.id),
        type=event.event_type,
        timestamp=event.created_at,
        headers={"traceparent": trace_parent} if trace_parent else None,
    )


async def relay_batch(
    factory: SessionFactory, exchange: AbstractExchange, *, batch_size: int
) -> int:
    """Publish up to ``batch_size`` pending events; returns how many the broker confirmed."""
    async with system_session(factory) as session, session.begin():
        events = (
            await session.scalars(
                select(OutboxEvent)
                .where(OutboxEvent.published_at.is_(None))
                .order_by(OutboxEvent.created_at)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            )
        ).all()
        if not events:
            return 0

        # Confirms are awaited concurrently; the row locks are held for the duration, which is
        # what keeps another replica from claiming the same rows mid-flight.
        outcomes = await asyncio.gather(
            *(
                exchange.publish(build_message(event), routing_key=event.event_type, mandatory=True)
                for event in events
            ),
            return_exceptions=True,
        )
        published_at = datetime.now(UTC)
        published = 0
        for event, outcome in zip(events, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                event.attempts += 1
                event.last_error = repr(outcome)[:_MAX_ERROR_LENGTH]
                logger.warning(
                    "outbox_publish_failed",
                    event_id=str(event.id),
                    event_type=event.event_type,
                    attempts=event.attempts,
                    error=event.last_error,
                )
            else:
                event.published_at = published_at
                published += 1
        return published


async def run(settings: Settings, stop: asyncio.Event) -> None:
    engine = create_engine(settings)
    factory = create_session_factory(engine)
    try:
        connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        async with connection:
            # on_return_raises turns "no queue bound for this routing key" into a failure, so an
            # event is never marked published while nobody could have received it.
            channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
            exchange = await channel.declare_exchange(
                settings.events_exchange, aio_pika.ExchangeType.TOPIC, durable=True
            )
            logger.info("outbox_relay_started", exchange=settings.events_exchange)
            while not stop.is_set():
                published = await relay_batch(
                    factory, exchange, batch_size=settings.outbox_batch_size
                )
                if published:
                    logger.info("outbox_batch_published", published=published)
                # Poll again immediately while there is a full backlog; otherwise idle (or back
                # off after failures) until the next tick or shutdown.
                if published < settings.outbox_batch_size:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            stop.wait(), timeout=settings.outbox_poll_interval_seconds
                        )
    finally:
        await engine.dispose()
        logger.info("outbox_relay_stopped")


async def _main() -> None:
    settings = get_settings()
    configure_logging(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Not supported on Windows event loops; Ctrl+C still interrupts there.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    await run(settings, stop)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
