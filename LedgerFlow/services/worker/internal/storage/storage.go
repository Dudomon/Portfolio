// Package storage downloads uploaded documents from S3-compatible storage
// and verifies them against the checksum carried in the event.
package storage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

var (
	// ErrNotFound means the object referenced by the event does not exist.
	ErrNotFound = errors.New("storage: object not found")
	// ErrChecksumMismatch means the bytes in storage are not the bytes the
	// API accepted. Processing them would attach an extraction to the wrong
	// content, so the message must not be retried.
	ErrChecksumMismatch = errors.New("storage: sha256 mismatch")
	// ErrTooLarge means the object exceeds the configured size limit.
	ErrTooLarge = errors.New("storage: object exceeds size limit")
)

// Config configures the S3 client.
type Config struct {
	Endpoint        string // empty for AWS; set for MinIO (e.g. http://minio:9000)
	Region          string
	Bucket          string
	AccessKeyID     string // empty to use the default AWS credential chain
	SecretAccessKey string
	UsePathStyle    bool // MinIO serves buckets as path segments, not subdomains
	MaxObjectBytes  int64
}

// S3 fetches document bytes.
type S3 struct {
	client   *s3.Client
	bucket   string
	maxBytes int64
}

// New builds an S3 client from cfg.
func New(ctx context.Context, cfg Config) (*S3, error) {
	if cfg.Bucket == "" {
		return nil, errors.New("storage: bucket is required")
	}
	if cfg.MaxObjectBytes <= 0 {
		return nil, errors.New("storage: max object bytes must be positive")
	}

	opts := []func(*awsconfig.LoadOptions) error{awsconfig.WithRegion(cfg.Region)}
	if cfg.AccessKeyID != "" {
		opts = append(opts, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(cfg.AccessKeyID, cfg.SecretAccessKey, ""),
		))
	}
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("storage: load aws config: %w", err)
	}

	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		if cfg.Endpoint != "" {
			o.BaseEndpoint = aws.String(cfg.Endpoint)
		}
		o.UsePathStyle = cfg.UsePathStyle
		// We verify the full-object SHA-256 ourselves. The SDK default
		// (validate when supported) adds per-part checksum work and logs a
		// warning against stores that return no flexible checksum.
		o.ResponseChecksumValidation = aws.ResponseChecksumValidationWhenRequired
	})
	return &S3{client: client, bucket: cfg.Bucket, maxBytes: cfg.MaxObjectBytes}, nil
}

// Fetch downloads key and verifies its SHA-256 equals wantSHA256 (hex). The
// hash is computed while streaming so the body is read exactly once.
func (s *S3) Fetch(ctx context.Context, key, wantSHA256 string) ([]byte, error) {
	want, err := hex.DecodeString(wantSHA256)
	if err != nil || len(want) != sha256.Size {
		return nil, fmt.Errorf("%w: expected digest %q is not a hex sha256", ErrChecksumMismatch, wantSHA256)
	}

	out, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		var noSuchKey *types.NoSuchKey
		if errors.As(err, &noSuchKey) {
			return nil, fmt.Errorf("%w: %s", ErrNotFound, key)
		}
		return nil, fmt.Errorf("storage: get object %s: %w", key, err)
	}
	defer out.Body.Close()

	if out.ContentLength != nil && *out.ContentLength > s.maxBytes {
		return nil, fmt.Errorf("%w: %d bytes > %d", ErrTooLarge, *out.ContentLength, s.maxBytes)
	}

	// Read one byte past the limit so an object without Content-Length is
	// still caught instead of silently truncated.
	h := sha256.New()
	data, err := io.ReadAll(io.TeeReader(io.LimitReader(out.Body, s.maxBytes+1), h))
	if err != nil {
		return nil, fmt.Errorf("storage: read object %s: %w", key, err)
	}
	if int64(len(data)) > s.maxBytes {
		return nil, fmt.Errorf("%w: more than %d bytes", ErrTooLarge, s.maxBytes)
	}
	if got := h.Sum(nil); !bytes.Equal(got, want) {
		return nil, fmt.Errorf("%w: got %x, want %x", ErrChecksumMismatch, got, want)
	}
	return data, nil
}
