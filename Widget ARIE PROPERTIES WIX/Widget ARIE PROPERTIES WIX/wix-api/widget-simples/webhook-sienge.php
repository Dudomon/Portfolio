<?php
/**
 * Webhook para automação de clientes Sienge -> Wix
 * Recebe notificações da Sienge, atualiza coleção Wix e envia email
 */

// Carregar configurações e helpers
$config = require_once __DIR__ . '/config.php';
require_once __DIR__ . '/wix-email-helper.php';

// Headers CORS
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, GET, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization');
header('Content-Type: application/json');

// Responder OPTIONS para CORS preflight
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

// Log de debug
function logDebug($message) {
    global $config;
    if ($config['logging']['enabled']) {
        $timestamp = date('Y-m-d H:i:s');
        file_put_contents($config['logging']['file'], "[$timestamp] $message\n", FILE_APPEND);
        
        // Rotacionar log se ficar muito grande
        if (file_exists($config['logging']['file']) && 
            filesize($config['logging']['file']) > $config['logging']['max_size']) {
            rename($config['logging']['file'], $config['logging']['file'] . '.old');
        }
    }
}

// Função para gerar senha simples
function gerarSenha($tamanho = null) {
    global $config;
    if ($tamanho === null) {
        $tamanho = $config['password']['length'];
    }
    $caracteres = $config['password']['characters'];
    $senha = '';
    for ($i = 0; $i < $tamanho; $i++) {
        $senha .= $caracteres[rand(0, strlen($caracteres) - 1)];
    }
    return $senha;
}

// Função para fazer requisição à API Wix
function wixApiRequest($endpoint, $method = 'GET', $data = null) {
    global $config;
    
    $url = $config['wix']['api_base_url'] . "/$endpoint";
    
    $headers = [
        'Authorization: Bearer ' . $config['wix']['api_key'],
        'Content-Type: application/json',
        'wix-site-id: ' . $config['wix']['site_id']
    ];
    
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $method);
    
    if ($data && ($method === 'POST' || $method === 'PUT')) {
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
    }
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    logDebug("Wix API Request: $method $url - Response Code: $httpCode - Response: " . substr($response, 0, 500));
    
    return [
        'success' => $httpCode >= 200 && $httpCode < 300,
        'data' => json_decode($response, true),
        'http_code' => $httpCode
    ];
}

// Função para enviar email (usa helper externo)
function enviarEmailCliente($destinatario, $nomeCliente, $cpfCnpj, $senha) {
    global $config;
    
    try {
        // Tentar enviar via API Wix primeiro
        $resultadoWix = enviarEmailViaWix($destinatario, $nomeCliente, $cpfCnpj, $senha, $config);
        
        if ($resultadoWix['success']) {
            logDebug("Email enviado via Wix API para: $destinatario - Cliente: $nomeCliente");
            return true;
        } else {
            logDebug("Falha na Wix API, tentando SMTP fallback...");
            // Fallback para SMTP tradicional
            return enviarEmailSMTP($destinatario, $nomeCliente, $cpfCnpj, $senha, $config);
        }
    } catch (Exception $e) {
        logDebug("Erro no envio de email: " . $e->getMessage());
        // Tentar SMTP como fallback
        return enviarEmailSMTP($destinatario, $nomeCliente, $cpfCnpj, $senha, $config);
    }
}

try {
    // Receber dados do webhook
    if ($_SERVER['REQUEST_METHOD'] === 'POST') {
        $input = file_get_contents('php://input');
        $webhookData = json_decode($input, true);
    } else if ($_SERVER['REQUEST_METHOD'] === 'GET') {
        // Para testes via GET, usar dados de exemplo
        $webhookData = [
            'nome' => 'THAIS CRISTINA JULIO BASTOS',
            'cpf_cnpj' => '374.554.078-66',
            'email' => 'thais.teste@email.com', 
            'telefone' => '(11) 99999-9999',
            'evento' => 'cliente_criado'
        ];
        logDebug("Teste via GET iniciado");
    } else {
        throw new Exception('Método não permitido');
    }
    
    if (!$webhookData) {
        throw new Exception('Dados inválidos recebidos');
    }
    
    logDebug("Webhook recebido: " . json_encode($webhookData));
    
    // Extrair dados do cliente (ajuste conforme estrutura da Sienge)
    $nomeCliente = $webhookData['nome'] ?? $webhookData['client_name'] ?? 'Cliente';
    $cpfCnpj = $webhookData['cpf_cnpj'] ?? $webhookData['document'] ?? '';
    $email = $webhookData['email'] ?? '';
    $telefone = $webhookData['telefone'] ?? $webhookData['phone'] ?? '';
    
    if (empty($cpfCnpj)) {
        throw new Exception('CPF/CNPJ não informado no webhook');
    }
    
    // Gerar senha
    $senhaGerada = gerarSenha();
    
    // Verificar se cliente já existe na coleção Wix
    $collectionId = $config['wix']['collection_id'];
    $clienteExistente = wixApiRequest("$collectionId/query", 'POST', [
        'filter' => [
            'cpfOuCnpj' => ['$eq' => $cpfCnpj]
        ]
    ]);
    
    $clienteData = [
        'cpfOuCnpj' => $cpfCnpj,
        'nome' => $nomeCliente,
        'email' => $email,
        'telefone' => $telefone,
        'senha' => $senhaGerada,
        'dataCreacao' => date('c'),
        'ativo' => true
    ];
    
    if ($clienteExistente['success'] && !empty($clienteExistente['data']['items'])) {
        // Cliente existe - atualizar
        $clienteId = $clienteExistente['data']['items'][0]['_id'];
        $resultado = wixApiRequest("$collectionId/items/$clienteId", 'PUT', [
            'item' => $clienteData
        ]);
        logDebug("Cliente atualizado: $cpfCnpj");
    } else {
        // Cliente novo - criar
        $resultado = wixApiRequest("$collectionId/items", 'POST', [
            'item' => $clienteData
        ]);
        logDebug("Novo cliente criado: $cpfCnpj");
    }
    
    if (!$resultado['success']) {
        throw new Exception('Erro ao salvar cliente na coleção Wix');
    }
    
    // Enviar email se email foi fornecido
    if (!empty($email)) {
        $emailEnviado = enviarEmailCliente($email, $nomeCliente, $cpfCnpj, $senhaGerada);
        if (!$emailEnviado) {
            logDebug("Erro ao enviar email para: $email");
        }
    }
    
    // Resposta de sucesso
    echo json_encode([
        'success' => true,
        'message' => 'Cliente processado com sucesso',
        'cliente' => [
            'nome' => $nomeCliente,
            'cpfCnpj' => $cpfCnpj,
            'email' => $email,
            'senhaGerada' => $senhaGerada
        ]
    ]);
    
} catch (Exception $e) {
    logDebug("Erro: " . $e->getMessage());
    
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => $e->getMessage()
    ]);
}
?>