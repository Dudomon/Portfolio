// Package config loads worker configuration from the environment.
package config

import (
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"time"

	"github.com/caarlos0/env/v11"
)

// Config is the complete worker configuration.
type Config struct {
	LogLevel        slog.Level    `env:"LOG_LEVEL" envDefault:"INFO"`
	MetricsAddr     string        `env:"METRICS_ADDR" envDefault:":9090"`
	ShutdownTimeout time.Duration `env:"SHUTDOWN_TIMEOUT" envDefault:"30s"`

	AMQPURL        string          `env:"AMQP_URL,required,notEmpty"`
	Concurrency    int             `env:"WORKER_CONCURRENCY" envDefault:"4"`
	Prefetch       int             `env:"WORKER_PREFETCH" envDefault:"8"`
	HandlerTimeout time.Duration   `env:"WORKER_HANDLER_TIMEOUT" envDefault:"5m"`
	MaxRetries     int             `env:"WORKER_MAX_RETRIES" envDefault:"3"`
	RetryDelays    []time.Duration `env:"WORKER_RETRY_DELAYS" envDefault:"5s,30s,2m" envSeparator:","`
	DedupCacheSize int             `env:"IDEMPOTENCY_CACHE_SIZE" envDefault:"10000"`

	S3  S3Config  `envPrefix:"S3_"`
	API APIConfig `envPrefix:"API_"`
	LLM LLMConfig `envPrefix:"LLM_"`
}

// S3Config configures document storage.
type S3Config struct {
	Endpoint        string `env:"ENDPOINT"`
	Region          string `env:"REGION" envDefault:"us-east-1"`
	Bucket          string `env:"BUCKET,required,notEmpty"`
	AccessKeyID     string `env:"ACCESS_KEY_ID"`
	SecretAccessKey string `env:"SECRET_ACCESS_KEY"`
	UsePathStyle    bool   `env:"USE_PATH_STYLE" envDefault:"true"`
	MaxObjectBytes  int64  `env:"MAX_OBJECT_BYTES" envDefault:"15728640"` // 15 MiB, the API's upload limit
}

// APIConfig configures calls to the LedgerFlow API.
type APIConfig struct {
	BaseURL      string        `env:"BASE_URL,required,notEmpty"` // e.g. http://api:8000/api/v1
	ServiceToken string        `env:"SERVICE_TOKEN,required,notEmpty"`
	Timeout      time.Duration `env:"TIMEOUT" envDefault:"10s"`
	MaxAttempts  int           `env:"MAX_ATTEMPTS" envDefault:"4"`
}

// LLMConfig configures the model provider.
type LLMConfig struct {
	BaseURL     string        `env:"BASE_URL" envDefault:"https://api.openai.com/v1"`
	APIKey      string        `env:"API_KEY,required,notEmpty"`
	Model       string        `env:"MODEL" envDefault:"gpt-4o-mini-2024-07-18"`
	Timeout     time.Duration `env:"TIMEOUT" envDefault:"60s"`
	MaxAttempts int           `env:"MAX_ATTEMPTS" envDefault:"3"`
	// Prices overrides the built-in price table: "model=in/out,..." in USD
	// per million tokens.
	Prices string `env:"PRICES"`
}

// Load parses and validates the environment.
func Load() (Config, error) {
	var cfg Config
	if err := env.Parse(&cfg); err != nil {
		return Config{}, fmt.Errorf("config: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, fmt.Errorf("config: %w", err)
	}
	return cfg, nil
}

// Validate checks cross-field invariants that struct tags cannot express.
func (c Config) Validate() error {
	var errs []error
	check := func(ok bool, format string, args ...any) {
		if !ok {
			errs = append(errs, fmt.Errorf(format, args...))
		}
	}

	check(c.Concurrency >= 1, "WORKER_CONCURRENCY must be >= 1, got %d", c.Concurrency)
	// A prefetch below concurrency leaves workers idle; the broker would
	// never hand out enough messages to keep them busy.
	check(c.Prefetch >= c.Concurrency, "WORKER_PREFETCH (%d) must be >= WORKER_CONCURRENCY (%d)", c.Prefetch, c.Concurrency)
	check(c.MaxRetries >= 0, "WORKER_MAX_RETRIES must be >= 0, got %d", c.MaxRetries)
	check(len(c.RetryDelays) > 0, "WORKER_RETRY_DELAYS must list at least one delay")
	for _, d := range c.RetryDelays {
		check(d > 0, "WORKER_RETRY_DELAYS entries must be positive, got %s", d)
	}
	check(c.DedupCacheSize >= 1, "IDEMPOTENCY_CACHE_SIZE must be >= 1, got %d", c.DedupCacheSize)
	check(c.ShutdownTimeout > 0, "SHUTDOWN_TIMEOUT must be positive")
	// The handler budget must cover at least one full model call, or every
	// PDF would time out and burn its retries.
	check(c.HandlerTimeout > c.LLM.Timeout, "WORKER_HANDLER_TIMEOUT (%s) must exceed LLM_TIMEOUT (%s)", c.HandlerTimeout, c.LLM.Timeout)

	check(c.S3.MaxObjectBytes > 0, "S3_MAX_OBJECT_BYTES must be positive")
	check((c.S3.AccessKeyID == "") == (c.S3.SecretAccessKey == ""), "S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set together")

	check(isHTTPURL(c.API.BaseURL), "API_BASE_URL %q is not an http(s) url", c.API.BaseURL)
	check(c.API.Timeout > 0, "API_TIMEOUT must be positive")
	check(c.API.MaxAttempts >= 1, "API_MAX_ATTEMPTS must be >= 1")

	check(isHTTPURL(c.LLM.BaseURL), "LLM_BASE_URL %q is not an http(s) url", c.LLM.BaseURL)
	check(c.LLM.Timeout > 0, "LLM_TIMEOUT must be positive")
	check(c.LLM.MaxAttempts >= 1, "LLM_MAX_ATTEMPTS must be >= 1")

	return errors.Join(errs...)
}

func isHTTPURL(raw string) bool {
	u, err := url.Parse(raw)
	return err == nil && (u.Scheme == "http" || u.Scheme == "https") && u.Host != ""
}
