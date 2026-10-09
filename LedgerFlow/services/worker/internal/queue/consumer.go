// Package queue consumes document events from RabbitMQ with manual
// acknowledgements, delayed retries through a TTL queue, and a dead-letter
// queue once the retry budget is spent.
//
// Delivery is at-least-once: a message is acknowledged only after its
// outcome (success, scheduled retry, or dead-letter copy) is durable, so a
// crash at any point leads to redelivery rather than loss. Handlers must be
// idempotent.
package queue

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	amqp "github.com/rabbitmq/amqp091-go"

	"github.com/Dudomon/ledgerflow/services/worker/internal/backoff"
)

// ErrPermanent marks failures that retrying cannot fix (malformed event,
// checksum mismatch, unsupported file). Handlers wrap it, e.g.
// fmt.Errorf("%w: %w", queue.ErrPermanent, err), to skip straight to the DLQ.
var ErrPermanent = errors.New("permanent failure")

// errFatal marks session failures that reconnecting cannot fix.
var errFatal = errors.New("unrecoverable amqp error")

const (
	heartbeat      = 10 * time.Second
	publishTimeout = 10 * time.Second
	maxErrorHeader = 1024
)

// Delivery is the broker-independent view of a message given to handlers.
type Delivery struct {
	MessageID   string
	Body        []byte
	Attempt     int // 1-based
	Redelivered bool
}

// Handler processes deliveries. Handle must be idempotent: the same message
// can arrive more than once.
type Handler interface {
	Handle(ctx context.Context, d Delivery) error
	// DeadLettered is called once the message has been moved to the DLQ, so
	// the owner can record the terminal failure.
	DeadLettered(ctx context.Context, d Delivery, cause error)
}

// Metrics receives per-message outcomes.
type Metrics interface {
	MessageProcessed(outcome string)
	MessageFailed(outcome string)
}

// Config configures the consumer.
type Config struct {
	URL            string
	ConsumerTag    string
	Concurrency    int // worker goroutines
	Prefetch       int // unacked messages the broker may push; >= Concurrency
	HandlerTimeout time.Duration
	DrainTimeout   time.Duration // how long in-flight work may finish on shutdown
	Retry          RetryPolicy
	Reconnect      backoff.Policy
}

// Consumer owns the AMQP connection lifecycle and a bounded worker pool.
type Consumer struct {
	cfg     Config
	handler Handler
	metrics Metrics
	log     *slog.Logger
	ready   atomic.Bool
}

// NewConsumer returns a consumer; call Run to start it.
func NewConsumer(cfg Config, h Handler, m Metrics, log *slog.Logger) *Consumer {
	return &Consumer{cfg: cfg, handler: h, metrics: m, log: log}
}

// Ready reports whether the consumer is currently attached to the broker.
func (c *Consumer) Ready() bool { return c.ready.Load() }

// Run consumes until ctx is canceled, reconnecting with backoff whenever the
// connection or a channel is lost. It returns nil on clean shutdown and an
// error only for failures that reconnecting cannot fix (bad credentials,
// conflicting topology); crashing surfaces those to the orchestrator.
func (c *Consumer) Run(ctx context.Context) error {
	failures := 0
	for {
		established, err := c.session(ctx)
		if ctx.Err() != nil {
			return nil
		}
		if errors.Is(err, errFatal) {
			return err
		}
		if established {
			failures = 0
		}
		delay := c.cfg.Reconnect.Delay(failures)
		failures++
		c.log.WarnContext(ctx, "amqp session ended; reconnecting", "error", err, "delay", delay)
		if err := backoff.Sleep(ctx, delay); err != nil {
			return nil
		}
	}
}

// session runs one connection until it fails or ctx is canceled. It reports
// whether consumption actually started, which resets the reconnect backoff.
func (c *Consumer) session(ctx context.Context) (bool, error) {
	conn, err := amqp.DialConfig(c.cfg.URL, amqp.Config{
		Heartbeat:  heartbeat,
		Locale:     "en_US",
		Properties: amqp.Table{"connection_name": c.cfg.ConsumerTag},
	})
	if err != nil {
		if errors.Is(err, amqp.ErrCredentials) {
			return false, fmt.Errorf("%w: dial: %w", errFatal, err)
		}
		return false, fmt.Errorf("dial: %w", err)
	}
	defer conn.Close()

	// Separate channels: publisher flow control on retries must not stall
	// acks on the consuming channel.
	consumeCh, err := conn.Channel()
	if err != nil {
		return false, fmt.Errorf("open consume channel: %w", err)
	}
	publishCh, err := conn.Channel()
	if err != nil {
		return false, fmt.Errorf("open publish channel: %w", err)
	}
	// Confirms make "retry scheduled" mean "retry stored by the broker"
	// before the original is acked.
	if err := publishCh.Confirm(false); err != nil {
		return false, fmt.Errorf("enable publisher confirms: %w", err)
	}
	if err := declareTopology(consumeCh); err != nil {
		// PRECONDITION_FAILED here means an existing queue or exchange was
		// declared with different arguments: a deployment conflict that
		// must be surfaced, not retried.
		var amqpErr *amqp.Error
		if errors.As(err, &amqpErr) && amqpErr.Code == amqp.PreconditionFailed {
			return false, fmt.Errorf("%w: %w", errFatal, err)
		}
		return false, err
	}
	if err := consumeCh.Qos(c.cfg.Prefetch, 0, false); err != nil {
		return false, fmt.Errorf("set qos: %w", err)
	}

	connClosed := conn.NotifyClose(make(chan *amqp.Error, 1))
	consumeClosed := consumeCh.NotifyClose(make(chan *amqp.Error, 1))
	publishClosed := publishCh.NotifyClose(make(chan *amqp.Error, 1))

	deliveries, err := consumeCh.Consume(MainQueue, c.cfg.ConsumerTag, false, false, false, false, nil)
	if err != nil {
		return false, fmt.Errorf("consume %s: %w", MainQueue, err)
	}

	// Handlers run on a context that survives shutdown so in-flight messages
	// can finish within DrainTimeout; cancelWork aborts them after that.
	workCtx, cancelWork := context.WithCancel(context.WithoutCancel(ctx))
	defer cancelWork()

	var (
		draining atomic.Bool
		wg       sync.WaitGroup
	)
	for range c.cfg.Concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for d := range deliveries {
				if draining.Load() {
					// Prefetched but not started: hand it back now so another
					// replica can take it instead of waiting for our restart.
					c.nack(workCtx, d, true)
					continue
				}
				c.process(workCtx, publishCh, d)
			}
		}()
	}
	workersDone := make(chan struct{})
	go func() {
		wg.Wait()
		close(workersDone)
	}()

	c.ready.Store(true)
	defer c.ready.Store(false)
	c.log.InfoContext(ctx, "consuming", "queue", MainQueue,
		"concurrency", c.cfg.Concurrency, "prefetch", c.cfg.Prefetch)

	var sessionErr error
	select {
	case <-ctx.Done():
		c.ready.Store(false)
		draining.Store(true)
		// Stop the broker pushing; the delivery channel closes once the
		// already-buffered messages have been handed to workers.
		if err := consumeCh.Cancel(c.cfg.ConsumerTag, false); err != nil {
			c.log.Warn("cancel consumer", "error", err)
		}
		timer := time.NewTimer(c.cfg.DrainTimeout)
		select {
		case <-workersDone:
			c.log.Info("drained in-flight messages")
		case <-timer.C:
			c.log.Warn("drain timeout; unacked messages will be redelivered", "timeout", c.cfg.DrainTimeout)
		}
		timer.Stop()
	case e := <-connClosed:
		sessionErr = closeError("connection", e)
	case e := <-consumeClosed:
		sessionErr = closeError("consume channel", e)
	case e := <-publishClosed:
		sessionErr = closeError("publish channel", e)
	case <-workersDone:
		sessionErr = errors.New("delivery channel closed by broker")
	}

	// Abort anything still running and close the connection, which closes
	// the delivery channel so idle workers exit. Unacked messages go back to
	// the queue; idempotent handling makes their redelivery harmless.
	cancelWork()
	_ = conn.Close()
	<-workersDone
	return true, sessionErr
}

func (c *Consumer) process(ctx context.Context, pub *amqp.Channel, d amqp.Delivery) {
	msg := Delivery{
		MessageID:   d.MessageId,
		Body:        d.Body,
		Attempt:     attemptFromHeaders(d.Headers),
		Redelivered: d.Redelivered,
	}
	log := c.log.With("message_id", msg.MessageID, "attempt", msg.Attempt)

	hctx, cancel := context.WithTimeout(ctx, c.cfg.HandlerTimeout)
	err := c.handler.Handle(hctx, msg)
	cancel()

	if err == nil {
		c.ack(ctx, d, log)
		c.metrics.MessageProcessed("success")
		return
	}
	if ctx.Err() != nil {
		// Shutdown or connection loss interrupted the handler; the failure
		// says nothing about the message. Leave it unacked for redelivery
		// without spending a retry.
		return
	}

	if delay, retry := c.cfg.Retry.Next(msg.Attempt); retry && !errors.Is(err, ErrPermanent) {
		if perr := c.publishRetry(ctx, pub, d, msg.Attempt+1, delay); perr != nil {
			log.ErrorContext(ctx, "scheduling retry failed; requeueing", "error", err, "publish_error", perr)
			c.nack(ctx, d, true)
			c.metrics.MessageFailed("requeued")
			return
		}
		c.ack(ctx, d, log)
		c.metrics.MessageFailed("retry")
		log.WarnContext(ctx, "handling failed; retry scheduled", "error", err, "delay", delay)
		return
	}

	if perr := c.publishDeadLetter(ctx, pub, d, msg.Attempt, err); perr != nil {
		// Fall back to broker-side dead-lettering via the main queue's DLX:
		// the message still reaches the DLQ, only without x-last-error.
		log.ErrorContext(ctx, "publishing to dlq failed; rejecting", "error", err, "publish_error", perr)
		c.nack(ctx, d, false)
	} else {
		c.ack(ctx, d, log)
	}
	c.metrics.MessageFailed("dead_letter")
	log.ErrorContext(ctx, "message dead-lettered", "error", err, "permanent", errors.Is(err, ErrPermanent))

	dctx, cancel := context.WithTimeout(ctx, c.cfg.HandlerTimeout)
	defer cancel()
	c.handler.DeadLettered(dctx, msg, err)
}

func (c *Consumer) publishRetry(ctx context.Context, ch *amqp.Channel, d amqp.Delivery, nextAttempt int, delay time.Duration) error {
	msg := republish(d)
	msg.Headers[attemptHeader] = int64(nextAttempt)
	// Per-message TTL: the broker dead-letters the copy back to the main
	// queue once it expires.
	msg.Expiration = strconv.FormatInt(delay.Milliseconds(), 10)
	return c.publish(ctx, ch, RetryQueue, msg)
}

func (c *Consumer) publishDeadLetter(ctx context.Context, ch *amqp.Channel, d amqp.Delivery, attempt int, cause error) error {
	msg := republish(d)
	msg.Headers[attemptHeader] = int64(attempt)
	msg.Headers[lastErrorHeader] = truncate(cause.Error(), maxErrorHeader)
	return c.publish(ctx, ch, DeadLetterQueue, msg)
}

// publish sends msg to queue via the default exchange and waits for the
// broker's confirm.
func (c *Consumer) publish(ctx context.Context, ch *amqp.Channel, queue string, msg amqp.Publishing) error {
	ctx, cancel := context.WithTimeout(ctx, publishTimeout)
	defer cancel()

	confirm, err := ch.PublishWithDeferredConfirmWithContext(ctx, "", queue, false, false, msg)
	if err != nil {
		return fmt.Errorf("publish to %s: %w", queue, err)
	}
	acked, err := confirm.WaitContext(ctx)
	if err != nil {
		return fmt.Errorf("await confirm from %s: %w", queue, err)
	}
	if !acked {
		return fmt.Errorf("broker nacked publish to %s", queue)
	}
	return nil
}

func (c *Consumer) ack(ctx context.Context, d amqp.Delivery, log *slog.Logger) {
	if err := d.Ack(false); err != nil {
		// The channel is gone; the broker will redeliver and the idempotent
		// handler will absorb the duplicate.
		log.WarnContext(ctx, "ack failed", "error", err)
	}
}

func (c *Consumer) nack(ctx context.Context, d amqp.Delivery, requeue bool) {
	if err := d.Nack(false, requeue); err != nil {
		c.log.WarnContext(ctx, "nack failed", "message_id", d.MessageId, "error", err)
	}
}

// republish copies a delivery into a new persistent publishing, keeping the
// original body byte-for-byte and its headers.
func republish(d amqp.Delivery) amqp.Publishing {
	headers := make(amqp.Table, len(d.Headers)+2)
	for k, v := range d.Headers {
		headers[k] = v
	}
	return amqp.Publishing{
		Headers:         headers,
		ContentType:     d.ContentType,
		ContentEncoding: d.ContentEncoding,
		DeliveryMode:    amqp.Persistent,
		Priority:        d.Priority,
		CorrelationId:   d.CorrelationId,
		MessageId:       d.MessageId,
		Timestamp:       d.Timestamp,
		Type:            d.Type,
		AppId:           d.AppId,
		Body:            d.Body,
	}
}

func closeError(what string, e *amqp.Error) error {
	if e == nil {
		return fmt.Errorf("%s closed", what)
	}
	return fmt.Errorf("%s closed: %w", what, e)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}
