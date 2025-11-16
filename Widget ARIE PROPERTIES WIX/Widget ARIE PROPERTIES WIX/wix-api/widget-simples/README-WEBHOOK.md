# 🚀 Sistema de Automação Sienge → Wix - ATUALIZADO

Sistema PHP para automatizar o gerenciamento de clientes entre Sienge e Wix com geração automática de senhas e envio de emails.

## 📝 HISTÓRICO DE DESENVOLVIMENTO - SESSÃO ATUAL

### ✅ Problemas Resolvidos:
1. **Campos da coleção ajustados**: `cpfCnpj` → `cpfOuCnpj` (conforme screenshot do Wix)
2. **API Key configurada**: Nova chave com todas as permissões
3. **Site ID confirmado**: `96359d97-c440-4f99-95be-77ba2f71c476`
4. **URLs corrigidas**: Sistema funcionando em `radioentrerios.com.br/wp-content/backends/`
5. **Método GET habilitado**: Para testes via navegador
6. **API v1 implementada**: Após testes mostrarem que v2 não funciona
7. **Logs detalhados**: Sistema de debug implementado

### 🐛 Problema Atual:
- **Erro 404 nas APIs Wix**: Pesquisa revelou que APIs só funcionam em **sites PUBLICADOS**
- **Site deve estar em modo LIVE** (não preview/draft) para APIs funcionarem

### 🔧 Arquivos Atualizados:
- `config.php`: Nova API Key + API v1 + Site ID correto
- `webhook-sienge.php`: Endpoints v1 + campo `cpfOuCnpj` + logs detalhados
- `debug-wix.php`: Script de teste da API
- `list-collections-v2.php`: Listagem de coleções (confirmou que "Cliente" existe)
- `view-logs.php`: Visualizador de logs

## 📋 Arquivos do Sistema

- `webhook-sienge.php` - Endpoint principal que recebe webhooks da Sienge
- `config.php` - Configurações centralizadas do sistema
- `wix-email-helper.php` - Helper para envio de emails via API Wix
- `test-webhook.php` - Script para testar o webhook
- `webhook-logs.txt` - Log de atividades (criado automaticamente)

## ⚙️ Configuração

### 1. Configurar credenciais no `config.php`

```php
'wix' => [
    'api_key' => 'SUA_WIX_API_KEY_AQUI',
    'site_id' => 'SEU_WIX_SITE_ID_AQUI',
    'collection_id' => 'Clientes',
    'api_base_url' => 'https://www.wixapis.com/wix-data/v1/collections'
],
```

### 2. Ajustar URL do portal no email

```php
'email' => [
    'portal_url' => 'https://seusite.wixsite.com/portal-cliente'
],
```

### 3. Upload dos arquivos para seu servidor

Suba todos os arquivos PHP para uma pasta acessível via web no seu servidor atual (`radioentrerios.com.br`).

### 4. Configurar webhook na Sienge

Configure na Sienge para enviar webhooks para:
```
https://www.radioentrerios.com.br/caminho/para/webhook-sienge.php
```

## 🧪 Como Testar

### 1. Testar localmente
```bash
php test-webhook.php
```

### 2. Testar via web
Acesse: `https://seudominio.com/test-webhook.php`

### 3. Verificar logs
Consulte o arquivo `webhook-logs.txt` para ver a atividade.

## 📊 Estrutura da Coleção Wix "Clientes"

O sistema criará/atualizará registros com esta estrutura:

```json
{
  "cpfCnpj": "374.554.078-66",
  "nome": "THAIS CRISTINA JULIO BASTOS", 
  "email": "cliente@email.com",
  "telefone": "(11) 99999-9999",
  "senha": "Abc123Xy", 
  "dataCreacao": "2025-08-02T10:30:00Z",
  "ativo": true
}
```

## 📧 Sistema de Email

O sistema tenta enviar emails nesta ordem:
1. **API Wix** (método preferido)
2. **SMTP tradicional** (fallback)

### Template do email inclui:
- 🎨 Design responsivo com cores da Arie Properties
- 📋 Dados de acesso (CPF/CNPJ e senha)
- 🔗 Link direto para o portal
- ✅ Lista de funcionalidades disponíveis

## 🔧 Webhook da Sienge

### Formato esperado do webhook:
```json
{
  "nome": "Nome do Cliente",
  "cpf_cnpj": "123.456.789-00", 
  "email": "cliente@email.com",
  "telefone": "(11) 99999-9999",
  "evento": "cliente_criado"
}
```

### Campos alternativos aceitos:
- `client_name` → `nome`
- `document` → `cpf_cnpj` 
- `phone` → `telefone`

## 🛡️ Segurança

- ✅ Logs detalhados de todas as operações
- ✅ Rotação automática de logs (10MB)
- ✅ Tratamento de erros robusto
- ✅ Fallback para múltiplos métodos de envio
- ✅ Validação de dados de entrada
- ✅ Headers CORS apropriados

## 📈 Monitoramento

### Verificar logs:
```bash
tail -f webhook-logs.txt
```

### Códigos de status importantes:
- `200` - Sucesso
- `400` - Dados inválidos 
- `500` - Erro interno

## 🔄 Fluxo Completo

1. **Sienge** adiciona novo cliente
2. **Webhook** é enviado para `webhook-sienge.php`
3. **Sistema** gera senha automaticamente
4. **Wix API** atualiza/cria registro na coleção "Clientes"
5. **Email** é enviado para o cliente com dados de acesso
6. **Log** registra toda a operação

## 🎯 Próximos Passos URGENTES

1. **PUBLICAR O SITE WIX** (modo LIVE) - APIs não funcionam em preview/draft
2. Testar webhook após publicação: `https://www.radioentrerios.com.br/wp-content/backends/webhook-sienge.php`
3. Verificar logs em: `https://www.radioentrerios.com.br/wp-content/backends/view-logs.php`
4. Configurar webhook na Sienge apontando para a URL acima
5. Testar com cliente real

## 🔍 URLs de Teste:
- **Webhook Principal**: `https://www.radioentrerios.com.br/wp-content/backends/webhook-sienge.php`
- **Debug API**: `https://www.radioentrerios.com.br/wp-content/backends/debug-wix.php`
- **Ver Logs**: `https://www.radioentrerios.com.br/wp-content/backends/view-logs.php`
- **Listar Coleções**: `https://www.radioentrerios.com.br/wp-content/backends/list-collections-v2.php`

## 📊 Status Atual:
- ✅ **Sistema completamente configurado**
- ✅ **Coleção "Cliente" confirmada no Wix**
- ✅ **API Key com todas as permissões**
- ❌ **Site precisa estar PUBLICADO para APIs funcionarem**

## 🔐 Credenciais Atuais:
- **API Key**: `IST.eyJraWQiOiJQb3pIX2FDMiIsImFsZyI6IlJTMjU2In0...` (Nova - 2025-08-04)
- **Site ID**: `96359d97-c440-4f99-95be-77ba2f71c476`
- **Collection ID**: `Cliente`
- **API Version**: v1 (v2 não funciona)

## 🚨 AÇÃO NECESSÁRIA:
**PUBLIQUE O SITE WIX EM MODO LIVE** - Essa é a única coisa impedindo o sistema de funcionar!

## 📞 Suporte

Em caso de problemas:
1. Verifique os logs em `webhook-logs.txt`
2. Teste cada componente individualmente
3. Valide as credenciais da API Wix
4. Confirme que a coleção "Clientes" existe no Wix