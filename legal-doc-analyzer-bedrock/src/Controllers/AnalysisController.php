<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Controllers;

use LegalDocAnalyzer\Core\Request;
use LegalDocAnalyzer\Core\Response;
use LegalDocAnalyzer\Services\BedrockService;
use LegalDocAnalyzer\Services\CacheService;
use LegalDocAnalyzer\Services\ObservabilityService;
use LegalDocAnalyzer\Services\SQSService;
use Ramsey\Uuid\Uuid;

class AnalysisController
{
    private BedrockService $bedrock;
    private CacheService $cache;
    private SQSService $sqs;
    private ObservabilityService $observability;

    public function __construct()
    {
        $this->observability = new ObservabilityService();
        $this->cache = new CacheService($this->observability);
        $this->bedrock = new BedrockService($this->cache, $this->observability);
        $this->sqs = new SQSService($this->observability);
    }

    public function extractEntities(Request $request): Response
    {
        $text = $request->getBodyParam('text');
        $documentType = $request->getBodyParam('document_type', 'outro');

        if (!$text) {
            return Response::error('Text is required', 400);
        }

        if (strlen($text) > 100000) {
            return $this->processAsync($request, 'extract_entities');
        }

        try {
            $result = $this->bedrock->extractEntities($text, $documentType);

            return Response::success([
                'entities' => $result,
                'document_type' => $documentType,
            ]);

        } catch (\Exception $e) {
            return Response::error('Entity extraction failed: ' . $e->getMessage(), 500);
        }
    }

    public function classify(Request $request): Response
    {
        $text = $request->getBodyParam('text');

        if (!$text) {
            return Response::error('Text is required', 400);
        }

        if (strlen($text) > 100000) {
            return $this->processAsync($request, 'classify');
        }

        try {
            $result = $this->bedrock->classify($text);

            return Response::success([
                'classification' => $result,
            ]);

        } catch (\Exception $e) {
            return Response::error('Classification failed: ' . $e->getMessage(), 500);
        }
    }

    public function summarize(Request $request): Response
    {
        $text = $request->getBodyParam('text');
        $maxLength = (int) $request->getBodyParam('max_length', 500);

        if (!$text) {
            return Response::error('Text is required', 400);
        }

        if (strlen($text) > 100000) {
            return $this->processAsync($request, 'summarize');
        }

        try {
            $result = $this->bedrock->summarize($text, $maxLength);

            return Response::success([
                'summary' => $result['summary'],
                'original_length' => strlen($text),
                'summary_length' => strlen($result['summary']),
                'tokens_used' => $result['usage'],
                'cost_usd' => $result['cost'],
            ]);

        } catch (\Exception $e) {
            return Response::error('Summarization failed: ' . $e->getMessage(), 500);
        }
    }

    public function compare(Request $request): Response
    {
        $text1 = $request->getBodyParam('text1');
        $text2 = $request->getBodyParam('text2');

        if (!$text1 || !$text2) {
            return Response::error('Both text1 and text2 are required', 400);
        }

        if (strlen($text1) + strlen($text2) > 100000) {
            return $this->processAsync($request, 'compare');
        }

        try {
            $prompt = <<<PROMPT
Compare the following two legal documents and provide:
1. Key similarities
2. Key differences
3. Conflicting clauses or statements
4. Overall similarity score (0-100)

Document 1:
{$text1}

Document 2:
{$text2}

Return a JSON object with: similarities (array), differences (array), conflicts (array), similarity_score (number)
PROMPT;

            $result = $this->bedrock->invoke($prompt, [
                'system' => 'You are an expert legal document comparator.',
            ]);

            $comparison = json_decode($result['content'], true) ?? [];

            return Response::success([
                'comparison' => $comparison,
                'tokens_used' => $result['usage'],
                'cost_usd' => $result['cost'],
            ]);

        } catch (\Exception $e) {
            return Response::error('Comparison failed: ' . $e->getMessage(), 500);
        }
    }

    public function getResult(Request $request): Response
    {
        $analysisId = $request->getParam('id');
        $result = $this->cache->get("analysis:{$analysisId}");

        if (!$result) {
            return Response::error('Analysis result not found', 404);
        }

        return Response::success($result);
    }

    public function getStatus(Request $request): Response
    {
        $analysisId = $request->getParam('id');
        $status = $this->cache->get("analysis_status:{$analysisId}");

        if (!$status) {
            return Response::error('Analysis not found', 404);
        }

        return Response::success([
            'analysis_id' => $analysisId,
            'status' => $status,
        ]);
    }

    private function processAsync(Request $request, string $operation): Response
    {
        $analysisId = Uuid::uuid4()->toString();
        $userId = $request->getAttribute('user')['id'];

        $this->cache->set("analysis_status:{$analysisId}", 'queued', 3600);

        $messageId = $this->sqs->sendMessage([
            'type' => 'analysis.' . $operation,
            'analysis_id' => $analysisId,
            'user_id' => $userId,
            'operation' => $operation,
            'data' => $request->getBody(),
            'priority' => 5,
        ]);

        return Response::created([
            'analysis_id' => $analysisId,
            'status' => 'queued',
            'message_id' => $messageId,
            'status_url' => "/api/v1/analysis/{$analysisId}/status",
            'result_url' => "/api/v1/analysis/{$analysisId}",
        ], 'Analysis queued for processing');
    }
}
