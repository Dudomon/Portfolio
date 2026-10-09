package queue

import (
	"fmt"

	amqp "github.com/rabbitmq/amqp091-go"
)

// Names from docs/CONTRACTS.md. Declarations are idempotent, but arguments
// must match whatever else declares the same entities, so they live here as
// constants rather than configuration.
const (
	EventsExchange     = "ledgerflow.events"
	DeadLetterExchange = "ledgerflow.dlx"
	MainQueue          = "worker.document-uploaded"
	RetryQueue         = "worker.document-uploaded.retry"
	DeadLetterQueue    = "worker.document-uploaded.dlq"
	RoutingKey         = "document.uploaded"
)

// declareTopology sets up:
//
//	ledgerflow.events --document.uploaded--> worker.document-uploaded
//	worker.document-uploaded.retry --(per-message TTL, DLX "" )--> worker.document-uploaded
//	ledgerflow.dlx --> worker.document-uploaded.dlq
//
// The retry queue has no consumers; messages sit there until their
// expiration and are then dead-lettered through the default exchange
// straight back to the main queue. Routing through ledgerflow.events instead
// would re-deliver the retry to every other subscriber of document.uploaded.
func declareTopology(ch *amqp.Channel) error {
	if err := ch.ExchangeDeclare(EventsExchange, amqp.ExchangeTopic, true, false, false, false, nil); err != nil {
		return fmt.Errorf("declare exchange %s: %w", EventsExchange, err)
	}
	if err := ch.ExchangeDeclare(DeadLetterExchange, amqp.ExchangeTopic, true, false, false, false, nil); err != nil {
		return fmt.Errorf("declare exchange %s: %w", DeadLetterExchange, err)
	}

	if _, err := ch.QueueDeclare(MainQueue, true, false, false, false, amqp.Table{
		"x-dead-letter-exchange": DeadLetterExchange,
	}); err != nil {
		return fmt.Errorf("declare queue %s: %w", MainQueue, err)
	}
	if err := ch.QueueBind(MainQueue, RoutingKey, EventsExchange, false, nil); err != nil {
		return fmt.Errorf("bind queue %s: %w", MainQueue, err)
	}

	if _, err := ch.QueueDeclare(RetryQueue, true, false, false, false, amqp.Table{
		"x-dead-letter-exchange":    "",
		"x-dead-letter-routing-key": MainQueue,
	}); err != nil {
		return fmt.Errorf("declare queue %s: %w", RetryQueue, err)
	}

	if _, err := ch.QueueDeclare(DeadLetterQueue, true, false, false, false, nil); err != nil {
		return fmt.Errorf("declare queue %s: %w", DeadLetterQueue, err)
	}
	// Broker-side dead-lettering keeps the routing key the message last had:
	// document.uploaded for first deliveries, the main queue's name for
	// messages that came back from the retry queue. Bind both.
	for _, key := range []string{RoutingKey, MainQueue} {
		if err := ch.QueueBind(DeadLetterQueue, key, DeadLetterExchange, false, nil); err != nil {
			return fmt.Errorf("bind queue %s to %s: %w", DeadLetterQueue, key, err)
		}
	}
	return nil
}
