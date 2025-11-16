<?php
/**
 * Script para listar todas as coleções do Wix
 */

$config = require_once __DIR__ . '/config.php';

$headers = [
    'Authorization: Bearer ' . $config['wix']['api_key'],
    'Content-Type: application/json',
    'wix-site-id: ' . $config['wix']['site_id']
];

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, 'https://www.wixapis.com/wix-data/v1/collections');
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

header('Content-Type: application/json');
echo json_encode([
    'http_code' => $httpCode,
    'collections' => json_decode($response, true)
], JSON_PRETTY_PRINT);
?>