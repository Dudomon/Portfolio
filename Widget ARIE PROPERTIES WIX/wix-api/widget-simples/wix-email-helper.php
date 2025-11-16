<?php
/**
 * Helper para envio de emails via API Wix
 */

/**
 * Envia email via API Wix
 */
function enviarEmailViaWix($destinatario, $nomeCliente, $cpfCnpj, $senha, $config) {
    $emailData = [
        'to' => [
            [
                'email' => $destinatario,
                'name' => $nomeCliente
            ]
        ],
        'subject' => $config['email']['subject'],
        'from' => [
            'email' => $config['email']['from_email'],
            'name' => $config['email']['from_name']
        ],
        'htmlContent' => gerarTemplateEmail($nomeCliente, $cpfCnpj, $senha, $config),
        'textContent' => gerarTemplateEmailTexto($nomeCliente, $cpfCnpj, $senha, $config)
    ];
    
    $url = 'https://www.wixapis.com/email/v1/send';
    
    $headers = [
        'Authorization: Bearer ' . $config['wix']['api_key'],
        'Content-Type: application/json',
        'wix-site-id: ' . $config['wix']['site_id']
    ];
    
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($emailData));
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    logDebug("Email API Response: HTTP $httpCode - " . substr($response, 0, 200));
    
    return [
        'success' => $httpCode >= 200 && $httpCode < 300,
        'data' => json_decode($response, true),
        'http_code' => $httpCode
    ];
}

/**
 * Gera template HTML do email
 */
function gerarTemplateEmail($nomeCliente, $cpfCnpj, $senha, $config) {
    $portalUrl = $config['email']['portal_url'];
    
    return "
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: #F7A81B; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }
            .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 5px 5px; }
            .credentials { background: white; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #F7A81B; }
            .btn { display: inline-block; background: #F7A81B; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; margin: 15px 0; }
            .footer { text-align: center; margin-top: 30px; font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <div class='container'>
            <div class='header'>
                <h1>🏠 Bem-vindo à Arie Properties!</h1>
            </div>
            
            <div class='content'>
                <h2>Olá, $nomeCliente!</h2>
                
                <p>É com grande satisfação que te damos as boas-vindas à família Arie Properties! 🎉</p>
                
                <p>Seus dados de acesso ao portal do cliente foram criados e estão prontos para uso:</p>
                
                <div class='credentials'>
                    <h3>📋 Seus dados de acesso:</h3>
                    <p><strong>CPF/CNPJ:</strong> $cpfCnpj</p>
                    <p><strong>Senha:</strong> $senha</p>
                </div>
                
                <p>Com seu acesso você poderá:</p>
                <ul>
                    <li>✅ Consultar seus boletos em aberto</li>
                    <li>💳 Gerar segunda via de boletos</li>
                    <li>📊 Acompanhar seu histórico de pagamentos</li>
                    <li>📞 Entrar em contato conosco</li>
                </ul>
                
                <div style='text-align: center;'>
                    <a href='$portalUrl' class='btn'>🔐 Acessar Portal do Cliente</a>
                </div>
                
                <p><strong>Importante:</strong> Guarde bem esses dados! Recomendamos que altere sua senha no primeiro acesso.</p>
            </div>
            
            <div class='footer'>
                <p>Esta é uma mensagem automática. Em caso de dúvidas, entre em contato conosco.</p>
                <p><strong>Arie Properties</strong> - Realizando sonhos, construindo o futuro 🏗️</p>
            </div>
        </div>
    </body>
    </html>
    ";
}

/**
 * Gera template de texto simples do email
 */
function gerarTemplateEmailTexto($nomeCliente, $cpfCnpj, $senha, $config) {
    $portalUrl = $config['email']['portal_url'];
    
    return "
🏠 BEM-VINDO À ARIE PROPERTIES! 🏠

Olá $nomeCliente!

É com grande satisfação que te damos as boas-vindas à família Arie Properties!

📋 SEUS DADOS DE ACESSO:
CPF/CNPJ: $cpfCnpj  
Senha: $senha

🔐 ACESSE SEU PORTAL: $portalUrl

COM SEU ACESSO VOCÊ PODERÁ:
✅ Consultar seus boletos em aberto
💳 Gerar segunda via de boletos  
📊 Acompanhar seu histórico de pagamentos
📞 Entrar em contato conosco

IMPORTANTE: Guarde bem esses dados! Recomendamos que altere sua senha no primeiro acesso.

Esta é uma mensagem automática. Em caso de dúvidas, entre em contato conosco.

ARIE PROPERTIES - Realizando sonhos, construindo o futuro 🏗️
    ";
}

/**
 * Fallback: Envio via SMTP tradicional (caso a API Wix falhe)
 */
function enviarEmailSMTP($destinatario, $nomeCliente, $cpfCnpj, $senha, $config) {
    $assunto = $config['email']['subject'];
    $mensagem = gerarTemplateEmailTexto($nomeCliente, $cpfCnpj, $senha, $config);
    
    $headers = [
        'From: ' . $config['email']['from_name'] . ' <' . $config['email']['from_email'] . '>',
        'Reply-To: ' . $config['email']['from_email'],
        'Content-Type: text/plain; charset=utf-8',
        'X-Mailer: PHP/' . phpversion()
    ];
    
    $sucesso = mail($destinatario, $assunto, $mensagem, implode("\r\n", $headers));
    
    logDebug("SMTP Email enviado para $destinatario: " . ($sucesso ? 'SUCESSO' : 'FALHOU'));
    
    return $sucesso;
}
?>