<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Services;

use Aws\Sqs\SqsClient;
use Aws\Exception\AwsException;
use LegalDocAnalyzer\Services\ObservabilityService;

class SQSService
{
    private SqsClient $client;
    private ObservabilityService $observability;
    private string $queueUrl;
    private string $dlqUrl;

    public function __construct(ObservabilityService $observability)
    {
        $this->client = new SqsClient([
            'region' => $_ENV['AWS_REGION'],
            'version' => 'latest',
            'credentials' => [
                'key' => $_ENV['AWS_ACCESS_KEY_ID'],
                'secret' => $_ENV['AWS_SECRET_ACCESS_KEY'],
            ],
        ]);

        $this->observability = $observability;
        $this->queueUrl = $_ENV['SQS_QUEUE_URL'];
        $this->dlqUrl = $_ENV['SQS_DLQ_URL'];
    }

    public function sendMessage(array $messageData, string $messageGroupId = 'default'): ?string
    {
        try {
            $this->observability->startTrace('sqs.send_message');

            $result = $this->client->sendMessage([
                'QueueUrl' => $this->queueUrl,
                'MessageBody' => json_encode($messageData),
                'MessageAttributes' => [
                    'Type' => [
                        'DataType' => 'String',
                        'StringValue' => $messageData['type'] ?? 'unknown',
                    ],
                    'Priority' => [
                        'DataType' => 'Number',
                        'StringValue' => (string) ($messageData['priority'] ?? 0),
                    ],
                ],
            ]);

            $messageId = $result->get('MessageId');

            $this->observability->recordMetric('sqs.message.sent', 1);
            $this->observability->log('info', 'Message sent to SQS', [
                'message_id' => $messageId,
                'type' => $messageData['type'] ?? 'unknown',
            ]);

            $this->observability->endTrace('sqs.send_message');

            return $messageId;

        } catch (AwsException $e) {
            $this->observability->recordMetric('sqs.message.error', 1);
            $this->observability->log('error', 'Failed to send message to SQS', [
                'error' => $e->getMessage(),
                'error_code' => $e->getAwsErrorCode(),
            ]);
            $this->observability->endTrace('sqs.send_message');

            return null;
        }
    }

    public function receiveMessages(int $maxMessages = 10, int $waitTime = 20): array
    {
        try {
            $result = $this->client->receiveMessage([
                'QueueUrl' => $this->queueUrl,
                'MaxNumberOfMessages' => $maxMessages,
                'WaitTimeSeconds' => $waitTime,
                'AttributeNames' => ['All'],
                'MessageAttributeNames' => ['All'],
            ]);

            $messages = $result->get('Messages') ?? [];

            $this->observability->recordMetric('sqs.messages.received', count($messages));

            return array_map(function ($message) {
                return [
                    'id' => $message['MessageId'],
                    'receipt_handle' => $message['ReceiptHandle'],
                    'body' => json_decode($message['Body'], true),
                    'attributes' => $message['Attributes'] ?? [],
                    'message_attributes' => $message['MessageAttributes'] ?? [],
                ];
            }, $messages);

        } catch (AwsException $e) {
            $this->observability->log('error', 'Failed to receive messages from SQS', [
                'error' => $e->getMessage(),
            ]);

            return [];
        }
    }

    public function deleteMessage(string $receiptHandle): bool
    {
        try {
            $this->client->deleteMessage([
                'QueueUrl' => $this->queueUrl,
                'ReceiptHandle' => $receiptHandle,
            ]);

            $this->observability->recordMetric('sqs.message.deleted', 1);

            return true;

        } catch (AwsException $e) {
            $this->observability->log('error', 'Failed to delete message from SQS', [
                'error' => $e->getMessage(),
            ]);

            return false;
        }
    }

    public function changeMessageVisibility(string $receiptHandle, int $visibilityTimeout): bool
    {
        try {
            $this->client->changeMessageVisibility([
                'QueueUrl' => $this->queueUrl,
                'ReceiptHandle' => $receiptHandle,
                'VisibilityTimeout' => $visibilityTimeout,
            ]);

            return true;

        } catch (AwsException $e) {
            $this->observability->log('error', 'Failed to change message visibility', [
                'error' => $e->getMessage(),
            ]);

            return false;
        }
    }

    public function sendBatch(array $messages): array
    {
        try {
            $entries = array_map(function ($message, $index) {
                return [
                    'Id' => (string) $index,
                    'MessageBody' => json_encode($message),
                    'MessageAttributes' => [
                        'Type' => [
                            'DataType' => 'String',
                            'StringValue' => $message['type'] ?? 'unknown',
                        ],
                    ],
                ];
            }, $messages, array_keys($messages));

            $result = $this->client->sendMessageBatch([
                'QueueUrl' => $this->queueUrl,
                'Entries' => $entries,
            ]);

            $successful = $result->get('Successful') ?? [];
            $failed = $result->get('Failed') ?? [];

            $this->observability->recordMetric('sqs.batch.sent', count($successful));
            $this->observability->recordMetric('sqs.batch.failed', count($failed));

            return [
                'successful' => count($successful),
                'failed' => count($failed),
                'failed_messages' => $failed,
            ];

        } catch (AwsException $e) {
            $this->observability->log('error', 'Failed to send batch messages', [
                'error' => $e->getMessage(),
            ]);

            return [
                'successful' => 0,
                'failed' => count($messages),
                'error' => $e->getMessage(),
            ];
        }
    }
}
