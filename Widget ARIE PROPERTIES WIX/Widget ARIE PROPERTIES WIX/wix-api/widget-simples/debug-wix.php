<?php
$config = require_once __DIR__ . '/config.php';

$headers = [
    'Authorization: Bearer ' . $config['wix']['api_key'],
    'Content-Type: application/json',
    'wix-site-id: ' . $config['wix']['site_id']
];

// Teste direto da API v1 
$data = [
    'cpfOuCnpj' => '123.456.789-00',
    'nome' => 'TESTE DIRETO',
    'email' => 'teste@email.com',
    'senha' => 'teste123'
];

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, 'https://www.wixapis.com/wix-data/v1/collections/Cliente/items');
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
curl_setopt($ch, CURLOPT_CUSTOMREQUEST, 'POST');
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

header('Content-Type: application/json');
echo json_encode([
    'url' => 'https://www.wixapis.com/wix-data/v1/collections/Cliente/items',
    'http_code' => $httpCode,
    'request' => $data,
    'response' => json_decode($response, true),
    'raw_response' => $response
], JSON_PRETTY_PRINT);
?>