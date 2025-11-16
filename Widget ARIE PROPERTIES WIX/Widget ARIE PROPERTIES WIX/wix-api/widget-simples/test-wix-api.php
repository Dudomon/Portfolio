<?php
$config = require_once __DIR__ . '/config.php';

$headers = [
    'Authorization: Bearer ' . $config['wix']['api_key'],
    'Content-Type: application/json',
    'wix-site-id: ' . $config['wix']['site_id']
];

// Testar diferentes endpoints
$endpoints = [
    'collections' => 'https://www.wixapis.com/wix-data/v1/collections',
    'collections-v2' => 'https://www.wixapis.com/wix-data/v2/collections',
    'items-direct' => 'https://www.wixapis.com/wix-data/v1/collections/Cliente/items'
];

header('Content-Type: application/json');

$results = [];
foreach ($endpoints as $name => $url) {
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    $results[$name] = [
        'url' => $url,
        'http_code' => $httpCode,
        'response' => substr($response, 0, 200)
    ];
}

echo json_encode($results, JSON_PRETTY_PRINT);
?>