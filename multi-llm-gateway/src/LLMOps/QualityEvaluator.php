<?php

declare(strict_types=1);

namespace MultiLLMGateway\LLMOps;

use MultiLLMGateway\Providers\LLMProviderInterface;
use MultiLLMGateway\Services\ObservabilityService;

class QualityEvaluator
{
    private ObservabilityService $observability;
    private LLMProviderInterface $judgeProvider;

    public function __construct(
        ObservabilityService $observability,
        LLMProviderInterface $judgeProvider
    ) {
        $this->observability = $observability;
        $this->judgeProvider = $judgeProvider;
    }

    public function evaluateResponse(string $prompt, string $response, array $criteria = []): array
    {
        $defaultCriteria = [
            'relevance' => 'Is the response relevant to the prompt?',
            'accuracy' => 'Is the information in the response accurate?',
            'completeness' => 'Does the response fully address the prompt?',
            'clarity' => 'Is the response clear and well-structured?',
        ];

        $evaluationCriteria = array_merge($defaultCriteria, $criteria);

        $evaluationPrompt = $this->buildEvaluationPrompt($prompt, $response, $evaluationCriteria);

        try {
            $this->observability->startTrace('quality.evaluation');

            $result = $this->judgeProvider->invoke($evaluationPrompt, [
                'max_tokens' => 1000,
                'temperature' => 0.3,
            ]);

            $evaluation = json_decode($result['content'], true) ?? [];

            $overallScore = $this->calculateOverallScore($evaluation);

            $this->observability->recordMetric('quality.score', $overallScore);
            $this->observability->log('info', 'Quality evaluation completed', [
                'overall_score' => $overallScore,
                'scores' => $evaluation,
            ]);

            $this->observability->endTrace('quality.evaluation');

            return [
                'overall_score' => $overallScore,
                'criteria_scores' => $evaluation,
                'provider' => $result['provider'],
                'cost' => $result['cost'],
            ];

        } catch (\Exception $e) {
            $this->observability->log('error', 'Quality evaluation failed', [
                'error' => $e->getMessage(),
            ]);

            return [
                'overall_score' => 0,
                'error' => $e->getMessage(),
            ];
        }
    }

    private function buildEvaluationPrompt(string $prompt, string $response, array $criteria): string
    {
        $criteriaText = '';

        foreach ($criteria as $criterion => $description) {
            $criteriaText .= "- {$criterion}: {$description}\n";
        }

        return <<<EVALUATION
Evaluate the following LLM response based on these criteria:

{$criteriaText}

Original Prompt:
{$prompt}

Response to Evaluate:
{$response}

Provide scores from 0-10 for each criterion and return ONLY a JSON object with this structure:
{
  "relevance": <score>,
  "accuracy": <score>,
  "completeness": <score>,
  "clarity": <score>
}

No additional text, only the JSON.
EVALUATION;
    }

    private function calculateOverallScore(array $scores): float
    {
        if (empty($scores)) {
            return 0.0;
        }

        $total = array_sum($scores);
        $count = count($scores);

        return round($total / $count, 2);
    }

    public function detectAnomalies(array $responses): array
    {
        $anomalies = [];

        foreach ($responses as $i => $response) {
            if ($response['latency'] > 10000) {
                $anomalies[] = [
                    'type' => 'high_latency',
                    'index' => $i,
                    'value' => $response['latency'],
                    'threshold' => 10000,
                ];
            }

            if ($response['cost'] > 0.10) {
                $anomalies[] = [
                    'type' => 'high_cost',
                    'index' => $i,
                    'value' => $response['cost'],
                    'threshold' => 0.10,
                ];
            }

            if (strlen($response['content']) < 10) {
                $anomalies[] = [
                    'type' => 'short_response',
                    'index' => $i,
                    'length' => strlen($response['content']),
                ];
            }

            if (!$response['success']) {
                $anomalies[] = [
                    'type' => 'failure',
                    'index' => $i,
                ];
            }
        }

        if (!empty($anomalies)) {
            $this->observability->recordMetric('quality.anomalies', count($anomalies));
            $this->observability->log('warning', 'Anomalies detected in responses', [
                'count' => count($anomalies),
                'anomalies' => $anomalies,
            ]);
        }

        return $anomalies;
    }

    public function compareProviders(string $prompt, array $providers): array
    {
        $results = [];

        foreach ($providers as $provider) {
            try {
                $response = $provider->invoke($prompt);
                $quality = $this->evaluateResponse($prompt, $response['content']);

                $results[$provider->getName()] = [
                    'response' => $response,
                    'quality' => $quality,
                    'score' => $quality['overall_score'],
                ];

            } catch (\Exception $e) {
                $results[$provider->getName()] = [
                    'error' => $e->getMessage(),
                    'score' => 0,
                ];
            }
        }

        usort($results, function ($a, $b) {
            return $b['score'] <=> $a['score'];
        });

        return $results;
    }
}
