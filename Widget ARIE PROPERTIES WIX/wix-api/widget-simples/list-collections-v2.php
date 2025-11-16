<?php
$config = require_once __DIR__ . '/config.php';

$headers = [
    'Authorization: Bearer ' . $config['wix']['api_key'],
    'Content-Type: application/json',
    'wix-site-id: ' . $config['wix']['site_id']
];

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, 'https://www.wixapis.com/wix-data/v2/collections');
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

$data = json_decode($response, true);

header('Content-Type: application/json');

// Filtrar apenas coleções do usuário (não Wix Apps)
$userCollections = [];
if (isset($data['collections'])) {
    foreach ($data['collections'] as $collection) {
        if ($collection['collectionType'] === 'USER') {
            $userCollections[] = [
                'id' => $collection['id'],
                'displayName' => $collection['displayName'] ?? $collection['id']
            ];
        }
    }
}

echo json_encode([
    'http_code' => $httpCode,
    'user_collections' => $userCollections,
    'all_collections' => $data['collections'] ?? []
], JSON_PRETTY_PRINT);
?>