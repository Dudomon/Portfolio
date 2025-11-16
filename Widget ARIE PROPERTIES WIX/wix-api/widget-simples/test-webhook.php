<?php
/**
 * Script para testar o webhook Sienge->Wix
 * Simula o envio de dados da Sienge para testar a automação
 */

// Dados de teste simulando webhook da Sienge (usando dados reais de teste)
$testData = [
    'nome' => 'THAIS CRISTINA JULIO BASTOS',
    'cpf_cnpj' => '374.554.078-66',
    'email' => 'thais.teste@email.com',
    'telefone' => '(11) 99999-9999',
    'evento' => 'cliente_criado',
    'timestamp' => date('c')
];

// URL do webhook (ajuste para sua URL)
$webhookUrl = 'http://localhost/webhook-sienge.php'; // Substitua pela URL real

echo "Testando webhook com dados:\n";
echo json_encode($testData, JSON_PRETTY_PRINT) . "\n\n";

// Enviar requisição POST
$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $webhookUrl);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($testData));
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Content-Type: application/json',
    'User-Agent: Sienge-Webhook-Test'
]);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_TIMEOUT, 30);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error = curl_error($ch);
curl_close($ch);

echo "Resposta do webhook:\n";
echo "HTTP Code: $httpCode\n";

if ($error) {
    echo "Erro cURL: $error\n";
} else {
    echo "Response: " . ($response ? $response : 'Sem resposta') . "\n";
}

// Verificar logs
$logFile = __DIR__ . '/webhook-logs.txt';
if (file_exists($logFile)) {
    echo "\nÚltimas linhas do log:\n";
    $logs = file($logFile);
    $lastLines = array_slice($logs, -5);
    foreach ($lastLines as $line) {
        echo $line;
    }
}
?>