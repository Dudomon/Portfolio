<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Services;

use Aws\S3\S3Client;
use Aws\Exception\AwsException;
use LegalDocAnalyzer\Services\ObservabilityService;

class S3Service
{
    private S3Client $client;
    private ObservabilityService $observability;
    private string $bucket;
    private string $kmsKeyId;

    public function __construct(ObservabilityService $observability)
    {
        $this->client = new S3Client([
            'region' => $_ENV['S3_BUCKET_REGION'],
            'version' => 'latest',
            'credentials' => [
                'key' => $_ENV['AWS_ACCESS_KEY_ID'],
                'secret' => $_ENV['AWS_SECRET_ACCESS_KEY'],
            ],
        ]);

        $this->observability = $observability;
        $this->bucket = $_ENV['S3_BUCKET_DOCUMENTS'];
        $this->kmsKeyId = $_ENV['KMS_KEY_ID'];
    }

    public function upload(string $key, string $content, array $metadata = []): bool
    {
        try {
            $this->observability->startTrace('s3.upload');

            $this->client->putObject([
                'Bucket' => $this->bucket,
                'Key' => $key,
                'Body' => $content,
                'ServerSideEncryption' => 'aws:kms',
                'SSEKMSKeyId' => $this->kmsKeyId,
                'Metadata' => $metadata,
                'ContentType' => $this->getContentType($key),
            ]);

            $this->observability->recordMetric('s3.upload.success', 1);
            $this->observability->recordMetric('s3.upload.size', strlen($content), 'Bytes');
            $this->observability->log('info', 'File uploaded to S3', [
                'bucket' => $this->bucket,
                'key' => $key,
                'size' => strlen($content),
            ]);

            $this->observability->endTrace('s3.upload');

            return true;

        } catch (AwsException $e) {
            $this->observability->recordMetric('s3.upload.error', 1);
            $this->observability->log('error', 'Failed to upload file to S3', [
                'bucket' => $this->bucket,
                'key' => $key,
                'error' => $e->getMessage(),
            ]);
            $this->observability->endTrace('s3.upload');

            return false;
        }
    }

    public function download(string $key): ?string
    {
        try {
            $this->observability->startTrace('s3.download');

            $result = $this->client->getObject([
                'Bucket' => $this->bucket,
                'Key' => $key,
            ]);

            $content = $result['Body']->getContents();

            $this->observability->recordMetric('s3.download.success', 1);
            $this->observability->recordMetric('s3.download.size', strlen($content), 'Bytes');

            $this->observability->endTrace('s3.download');

            return $content;

        } catch (AwsException $e) {
            $this->observability->recordMetric('s3.download.error', 1);
            $this->observability->log('error', 'Failed to download file from S3', [
                'bucket' => $this->bucket,
                'key' => $key,
                'error' => $e->getMessage(),
            ]);
            $this->observability->endTrace('s3.download');

            return null;
        }
    }

    public function delete(string $key): bool
    {
        try {
            $this->client->deleteObject([
                'Bucket' => $this->bucket,
                'Key' => $key,
            ]);

            $this->observability->recordMetric('s3.delete.success', 1);
            $this->observability->log('info', 'File deleted from S3', [
                'bucket' => $this->bucket,
                'key' => $key,
            ]);

            return true;

        } catch (AwsException $e) {
            $this->observability->recordMetric('s3.delete.error', 1);
            $this->observability->log('error', 'Failed to delete file from S3', [
                'bucket' => $this->bucket,
                'key' => $key,
                'error' => $e->getMessage(),
            ]);

            return false;
        }
    }

    public function exists(string $key): bool
    {
        try {
            return $this->client->doesObjectExist($this->bucket, $key);
        } catch (AwsException $e) {
            return false;
        }
    }

    public function getPresignedUrl(string $key, int $expiresIn = 3600): string
    {
        try {
            $cmd = $this->client->getCommand('GetObject', [
                'Bucket' => $this->bucket,
                'Key' => $key,
            ]);

            $request = $this->client->createPresignedRequest($cmd, "+{$expiresIn} seconds");

            return (string) $request->getUri();

        } catch (AwsException $e) {
            $this->observability->log('error', 'Failed to generate presigned URL', [
                'bucket' => $this->bucket,
                'key' => $key,
                'error' => $e->getMessage(),
            ]);

            return '';
        }
    }

    private function getContentType(string $filename): string
    {
        $extension = pathinfo($filename, PATHINFO_EXTENSION);

        $mimeTypes = [
            'pdf' => 'application/pdf',
            'doc' => 'application/msword',
            'docx' => 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'txt' => 'text/plain',
            'json' => 'application/json',
        ];

        return $mimeTypes[$extension] ?? 'application/octet-stream';
    }

    public function listObjects(string $prefix = '', int $maxKeys = 1000): array
    {
        try {
            $result = $this->client->listObjectsV2([
                'Bucket' => $this->bucket,
                'Prefix' => $prefix,
                'MaxKeys' => $maxKeys,
            ]);

            $objects = $result->get('Contents') ?? [];

            return array_map(function ($object) {
                return [
                    'key' => $object['Key'],
                    'size' => $object['Size'],
                    'last_modified' => $object['LastModified']->format('c'),
                ];
            }, $objects);

        } catch (AwsException $e) {
            $this->observability->log('error', 'Failed to list objects from S3', [
                'bucket' => $this->bucket,
                'prefix' => $prefix,
                'error' => $e->getMessage(),
            ]);

            return [];
        }
    }
}
