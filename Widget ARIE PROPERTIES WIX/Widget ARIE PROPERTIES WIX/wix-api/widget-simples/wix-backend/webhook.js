// webhook.js - Webhook Sienge->Wix em JavaScript
import {ok, serverError, badRequest} from 'wix-http-functions';
import wixData from 'wix-data';
import {send} from 'wix-email';

// Configurações (inline para evitar problemas de import)
const CONFIG = {
    wix: {
        collection_id: 'Cliente'
    },
    password: {
        length: 8,
        characters: 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    },
    email: {
        from_name: 'Arie Properties',
        from_email: 'noreply@arieproperties.com',
        subject: 'Bem-vindo à Arie Properties - Seus dados de acesso',
        portal_url: 'https://www.arieproperties.com/portal-cliente'
    },
    logging: {
        enabled: true,
        prefix: 'ARIE_WEBHOOK'
    }
};

// Função para gerar senha aleatória
function gerarSenha(tamanho) {
    if (!tamanho) {
        tamanho = CONFIG.password.length;
    }
    const caracteres = CONFIG.password.characters;
    let senha = '';
    for (let i = 0; i < tamanho; i++) {
        senha += caracteres[Math.floor(Math.random() * caracteres.length)];
    }
    return senha;
}

// Função para log de debug
function logDebug(message) {
    if (CONFIG.logging.enabled) {
        const timestamp = new Date().toISOString();
        console.log(`[${timestamp}] ${CONFIG.logging.prefix}: ${message}`);
    }
}

// Função para gerar template HTML do email
function gerarTemplateEmail(nomeCliente, cpfCnpj, senha) {
    const portalUrl = CONFIG.email.portal_url;
    
    return `
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
                <h2>Olá, ${nomeCliente}!</h2>
                
                <p>É com grande satisfação que te damos as boas-vindas à família Arie Properties! 🎉</p>
                
                <p>Seus dados de acesso ao portal do cliente foram criados e estão prontos para uso:</p>
                
                <div class='credentials'>
                    <h3>📋 Seus dados de acesso:</h3>
                    <p><strong>CPF/CNPJ:</strong> ${cpfCnpj}</p>
                    <p><strong>Senha:</strong> ${senha}</p>
                </div>
                
                <p>Com seu acesso você poderá:</p>
                <ul>
                    <li>✅ Consultar seus boletos em aberto</li>
                    <li>💳 Gerar segunda via de boletos</li>
                    <li>📊 Acompanhar seu histórico de pagamentos</li>
                    <li>📞 Entrar em contato conosco</li>
                </ul>
                
                <div style='text-align: center;'>
                    <a href='${portalUrl}' class='btn'>🔐 Acessar Portal do Cliente</a>
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
    `;
}

// Função para enviar email via Wix
async function enviarEmailCliente(destinatario, nomeCliente, cpfCnpj, senha) {
    try {
        const emailData = {
            to: destinatario,
            from: CONFIG.email.from_email,
            subject: CONFIG.email.subject,
            html: gerarTemplateEmail(nomeCliente, cpfCnpj, senha)
        };

        const result = await send(emailData);
        
        logDebug(`Email enviado para ${destinatario} - Cliente: ${nomeCliente}`);
        
        return {
            success: true,
            data: result
        };
        
    } catch (error) {
        logDebug(`Erro ao enviar email para ${destinatario}: ${error.message}`);
        
        return {
            success: false,
            error: error.message
        };
    }
}

// Webhook principal - recebe dados da Sienge
export async function post_webhook(request) {
    try {
        // Headers CORS
        const corsHeaders = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Content-Type': 'application/json'
        };

        // Receber dados do webhook
        let webhookData;
        
        if (request.body) {
            webhookData = request.body;
        } else {
            // Para testes, usar dados de exemplo
            webhookData = {
                nome: 'THAIS CRISTINA JULIO BASTOS',
                cpf_cnpj: '374.554.078-66',
                email: 'thais.teste@email.com',
                telefone: '(11) 99999-9999',
                evento: 'cliente_criado'
            };
            logDebug("Teste via GET iniciado");
        }
        
        if (!webhookData) {
            throw new Error('Dados inválidos recebidos');
        }
        
        logDebug("Webhook recebido: " + JSON.stringify(webhookData));
        
        // Extrair dados do cliente
        const nomeCliente = webhookData.nome || webhookData.client_name || 'Cliente';
        const cpfCnpj = webhookData.cpf_cnpj || webhookData.document || '';
        const email = webhookData.email || '';
        const telefone = webhookData.telefone || webhookData.phone || '';
        
        if (!cpfCnpj) {
            throw new Error('CPF/CNPJ não informado no webhook');
        }
        
        // Gerar senha
        const senhaGerada = gerarSenha();
        
        // Verificar se cliente já existe na coleção Wix
        const clienteExistente = await wixData.query(CONFIG.wix.collection_id)
            .eq('cpfOuCnpj', cpfCnpj)
            .find();
        
        const clienteData = {
            title: nomeCliente, // Campo Title do Wix
            nome: nomeCliente,
            email: email,
            cpfOuCnpj: cpfCnpj,
            senha: senhaGerada
        };
        
        let resultado;
        
        if (clienteExistente.items.length > 0) {
            // Cliente existe - atualizar
            const clienteId = clienteExistente.items[0]._id;
            resultado = await wixData.update(CONFIG.wix.collection_id, {
                _id: clienteId,
                ...clienteData
            });
            logDebug(`Cliente atualizado: ${cpfCnpj}`);
        } else {
            // Cliente novo - criar
            resultado = await wixData.save(CONFIG.wix.collection_id, clienteData);
            logDebug(`Novo cliente criado: ${cpfCnpj}`);
        }
        
        // Enviar email se email foi fornecido
        if (email) {
            const emailResult = await enviarEmailCliente(email, nomeCliente, cpfCnpj, senhaGerada);
            if (!emailResult.success) {
                logDebug(`Erro ao enviar email para: ${email} - ${emailResult.error}`);
            }
        }
        
        // Resposta de sucesso
        return ok({
            headers: corsHeaders,
            body: {
                success: true,
                message: 'Cliente processado com sucesso',
                cliente: {
                    nome: nomeCliente,
                    cpfCnpj: cpfCnpj,
                    email: email,
                    senhaGerada: senhaGerada
                }
            }
        });
        
    } catch (error) {
        logDebug(`Erro: ${error.message}`);
        
        return serverError({
            body: {
                success: false,
                error: error.message
            }
        });
    }
}

// Função OPTIONS para CORS preflight
export function options_webhook(request) {
    return ok({
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }
    });
}

// Função de teste GET
export function get_webhook(request) {
    // Redirecionar para POST com dados de teste
    return post_webhook(request);
}