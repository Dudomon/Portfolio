// Command worker consumes document.uploaded events, extracts invoice fields
// and reports the results to the LedgerFlow API.
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/Dudomon/ledgerflow/services/worker/internal/apiclient"
	"github.com/Dudomon/ledgerflow/services/worker/internal/backoff"
	"github.com/Dudomon/ledgerflow/services/worker/internal/config"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract/llm"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract/nfe"
	"github.com/Dudomon/ledgerflow/services/worker/internal/idempotency"
	"github.com/Dudomon/ledgerflow/services/worker/internal/observability"
	"github.com/Dudomon/ledgerflow/services/worker/internal/pipeline"
	"github.com/Dudomon/ledgerflow/services/worker/internal/queue"
	"github.com/Dudomon/ledgerflow/services/worker/internal/storage"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "worker:", err)
		os.Exit(1)
	}
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cfg, err := config.Load()
	if err != nil {
		return err
	}
	log := observability.NewLogger(os.Stdout, cfg.LogLevel)

	reg := observability.NewRegistry()
	metrics := observability.NewMetrics(reg)

	store, err := storage.New(ctx, storage.Config{
		Endpoint:        cfg.S3.Endpoint,
		Region:          cfg.S3.Region,
		Bucket:          cfg.S3.Bucket,
		AccessKeyID:     cfg.S3.AccessKeyID,
		SecretAccessKey: cfg.S3.SecretAccessKey,
		UsePathStyle:    cfg.S3.UsePathStyle,
		MaxObjectBytes:  cfg.S3.MaxObjectBytes,
	})
	if err != nil {
		return err
	}

	prices := llm.DefaultPrices()
	if cfg.LLM.Prices != "" {
		if prices, err = llm.ParsePrices(cfg.LLM.Prices); err != nil {
			return fmt.Errorf("LLM_PRICES: %w", err)
		}
	}
	model, err := llm.New(llm.Config{
		BaseURL:     cfg.LLM.BaseURL,
		APIKey:      cfg.LLM.APIKey,
		Model:       cfg.LLM.Model,
		Timeout:     cfg.LLM.Timeout,
		MaxAttempts: cfg.LLM.MaxAttempts,
		Backoff:     backoff.Policy{Base: time.Second, Max: 20 * time.Second},
		Prices:      prices,
	}, metrics, log)
	if err != nil {
		return err
	}

	api, err := apiclient.New(apiclient.Config{
		BaseURL:      cfg.API.BaseURL,
		ServiceToken: cfg.API.ServiceToken,
		MaxAttempts:  cfg.API.MaxAttempts,
		Backoff:      backoff.Policy{Base: 250 * time.Millisecond, Max: 5 * time.Second},
		HTTPClient: &http.Client{
			Timeout:   cfg.API.Timeout,
			Transport: observability.TracingTransport(http.DefaultTransport),
		},
	}, log)
	if err != nil {
		return err
	}

	processor := pipeline.New(
		store,
		extract.NewRouter(nfe.NewParser(), model, log),
		api,
		idempotency.NewLRU(cfg.DedupCacheSize),
		metrics,
		log,
	)

	hostname, err := os.Hostname()
	if err != nil {
		hostname = "unknown"
	}
	consumer := queue.NewConsumer(queue.Config{
		URL:            cfg.AMQPURL,
		ConsumerTag:    "ledgerflow-worker@" + hostname,
		Concurrency:    cfg.Concurrency,
		Prefetch:       cfg.Prefetch,
		HandlerTimeout: cfg.HandlerTimeout,
		DrainTimeout:   cfg.ShutdownTimeout,
		Retry:          queue.RetryPolicy{MaxRetries: cfg.MaxRetries, Delays: cfg.RetryDelays},
		Reconnect:      backoff.Policy{Base: time.Second, Max: 30 * time.Second},
	}, processor, metrics, log)

	srv := observability.NewServer(cfg.MetricsAddr, reg, consumer.Ready)

	log.Info("worker starting", "metrics_addr", cfg.MetricsAddr, "llm_model", cfg.LLM.Model,
		"concurrency", cfg.Concurrency, "max_retries", cfg.MaxRetries)

	g, gctx := errgroup.WithContext(ctx)
	g.Go(func() error { return consumer.Run(gctx) })
	g.Go(func() error { return observability.Serve(gctx, srv, cfg.ShutdownTimeout) })
	if err := g.Wait(); err != nil {
		return err
	}
	log.Info("worker stopped")
	return nil
}

// Compile-time checks that the concrete types satisfy the interfaces their
// consumers declare.
var (
	_ queue.Handler         = (*pipeline.Processor)(nil)
	_ queue.Metrics         = (*observability.Metrics)(nil)
	_ pipeline.Metrics      = (*observability.Metrics)(nil)
	_ pipeline.ObjectStore  = (*storage.S3)(nil)
	_ pipeline.API          = (*apiclient.Client)(nil)
	_ pipeline.Deduplicator = (*idempotency.LRU)(nil)
	_ extract.Extractor     = (*llm.Client)(nil)
	_ extract.Extractor     = (*nfe.Parser)(nil)
	_ llm.UsageRecorder     = (*observability.Metrics)(nil)
)
