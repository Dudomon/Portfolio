<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Controllers;

use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;
use LegalDocAnalyzer\Services\S3Service;
use LegalDocAnalyzer\Services\SQSService;
use LegalDocAnalyzer\Services\ObservabilityService;
use Ramsey\Uuid\Uuid;

class DocumentController
{
    private S3Service $s3;
    private SQSService $sqs;
    private ObservabilityService $observability;

    public function __construct()
    {
        $this->observability = new ObservabilityService();
        $this->s3 = new S3Service($this->observability);
        $this->sqs = new SQSService($this->observability);
    }

    public function upload(Request $request): Response
    {
        $file = $request->getFile('document');

        if (!$file) {
            return Response::error('No file uploaded', 400);
        }

        if ($file['error'] !== UPLOAD_ERR_OK) {
            return Response::error('File upload failed', 400);
        }

        $allowedTypes = ['application/pdf', 'application/msword',
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'text/plain'];

        if (!in_array($file['type'], $allowedTypes)) {
            return Response::error('Invalid file type. Only PDF, DOC, DOCX, and TXT are allowed', 400);
        }

        $maxSize = 10 * 1024 * 1024;
        if ($file['size'] > $maxSize) {
            return Response::error('File size exceeds 10MB limit', 400);
        }

        $documentId = Uuid::uuid4()->toString();
        $documentType = $request->getBodyParam('document_type', 'outro');
        $userId = $request->getAttribute('user')['id'];

        $key = "documents/{$userId}/{$documentId}/" . basename($file['name']);
        $content = file_get_contents($file['tmp_name']);

        $metadata = [
            'document-id' => $documentId,
            'document-type' => $documentType,
            'user-id' => $userId,
            'original-filename' => $file['name'],
            'upload-timestamp' => (string) time(),
        ];

        if (!$this->s3->upload($key, $content, $metadata)) {
            return Response::error('Failed to upload document', 500);
        }

        $messageId = $this->sqs->sendMessage([
            'type' => 'document.uploaded',
            'document_id' => $documentId,
            'document_type' => $documentType,
            'user_id' => $userId,
            's3_key' => $key,
            'filename' => $file['name'],
            'size' => $file['size'],
            'priority' => $documentType === 'sentenca' ? 10 : 5,
        ]);

        return Response::created([
            'document_id' => $documentId,
            'filename' => $file['name'],
            'size' => $file['size'],
            'document_type' => $documentType,
            's3_key' => $key,
            'processing_message_id' => $messageId,
        ], 'Document uploaded successfully');
    }

    public function get(Request $request): Response
    {
        $documentId = $request->getParam('id');
        $userId = $request->getAttribute('user')['id'];

        $key = "documents/{$userId}/{$documentId}/";
        $objects = $this->s3->listObjects($key);

        if (empty($objects)) {
            return Response::error('Document not found', 404);
        }

        $presignedUrl = $this->s3->getPresignedUrl($objects[0]['key'], 3600);

        return Response::success([
            'document_id' => $documentId,
            'key' => $objects[0]['key'],
            'size' => $objects[0]['size'],
            'last_modified' => $objects[0]['last_modified'],
            'download_url' => $presignedUrl,
            'expires_in' => 3600,
        ]);
    }

    public function list(Request $request): Response
    {
        $userId = $request->getAttribute('user')['id'];
        $prefix = "documents/{$userId}/";

        $objects = $this->s3->listObjects($prefix);

        $documents = [];
        foreach ($objects as $object) {
            $parts = explode('/', $object['key']);
            $documentId = $parts[2] ?? null;

            if ($documentId && !isset($documents[$documentId])) {
                $documents[$documentId] = [
                    'document_id' => $documentId,
                    'key' => $object['key'],
                    'size' => $object['size'],
                    'last_modified' => $object['last_modified'],
                ];
            }
        }

        return Response::success([
            'documents' => array_values($documents),
            'total' => count($documents),
        ]);
    }

    public function delete(Request $request): Response
    {
        $documentId = $request->getParam('id');
        $userId = $request->getAttribute('user')['id'];

        $prefix = "documents/{$userId}/{$documentId}/";
        $objects = $this->s3->listObjects($prefix);

        if (empty($objects)) {
            return Response::error('Document not found', 404);
        }

        $deleted = 0;
        foreach ($objects as $object) {
            if ($this->s3->delete($object['key'])) {
                $deleted++;
            }
        }

        return Response::success([
            'document_id' => $documentId,
            'files_deleted' => $deleted,
        ], 'Document deleted successfully');
    }
}
