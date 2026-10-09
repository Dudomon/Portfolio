package config

import (
	"log/slog"
	"slices"
	"strings"
	"testing"
	"time"
)

func setRequired(t *testing.T) {
	t.Helper()
	t.Setenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")
	t.Setenv("S3_BUCKET", "ledgerflow-documents")
	t.Setenv("API_BASE_URL", "http://localhost:8000/api/v1")
	t.Setenv("API_SERVICE_TOKEN", "dev-service-token")
	t.Setenv("LLM_API_KEY", "sk-test")
}

func TestLoadDefaults(t *testing.T) {
	setRequired(t)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.LogLevel != slog.LevelInfo || cfg.Concurrency != 4 || cfg.MaxRetries != 3 {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}
	if want := []time.Duration{5 * time.Second, 30 * time.Second, 2 * time.Minute}; !slices.Equal(cfg.RetryDelays, want) {
		t.Fatalf("RetryDelays = %v, want %v", cfg.RetryDelays, want)
	}
	if !cfg.S3.UsePathStyle || cfg.S3.MaxObjectBytes != 15<<20 {
		t.Fatalf("unexpected S3 defaults: %+v", cfg.S3)
	}
}

func TestLoadRejectsInvalid(t *testing.T) {
	tests := []struct {
		name    string
		env     map[string]string
		wantErr string
	}{
		{name: "prefetch below concurrency", env: map[string]string{"WORKER_CONCURRENCY": "8", "WORKER_PREFETCH": "4"}, wantErr: "WORKER_PREFETCH"},
		{name: "non-positive delay", env: map[string]string{"WORKER_RETRY_DELAYS": "5s,0s"}, wantErr: "WORKER_RETRY_DELAYS"},
		{name: "handler shorter than llm call", env: map[string]string{"WORKER_HANDLER_TIMEOUT": "30s"}, wantErr: "WORKER_HANDLER_TIMEOUT"},
		{name: "half of s3 credentials", env: map[string]string{"S3_ACCESS_KEY_ID": "minio"}, wantErr: "S3_SECRET_ACCESS_KEY"},
		{name: "api url without scheme", env: map[string]string{"API_BASE_URL": "api:8000"}, wantErr: "API_BASE_URL"},
		{name: "missing required", env: map[string]string{"LLM_API_KEY": ""}, wantErr: "LLM_API_KEY"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setRequired(t)
			for k, v := range tt.env {
				t.Setenv(k, v)
			}
			_, err := Load()
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Load error = %v, want mention of %s", err, tt.wantErr)
			}
		})
	}
}
