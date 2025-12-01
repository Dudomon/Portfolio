# 🧪 Como Testar o Chatbot

## ✅ Correções Aplicadas

### 1. **Correção da URL da API** ✓
- **Problema:** Caminho relativo causava erro 404
- **Solução:** Detecção automática da URL completa baseada no domínio
- **Arquivo:** `chatbot.js` linha 33-44

### 2. **Correção do Formato do Histórico** ✓
- **Problema:** Frontend enviava `{text: ...}` mas backend esperava `{content: ...}`
- **Solução:** Backend agora suporta ambos os formatos
- **Arquivo:** `groq_api.php` linha 75

### 3. **Integração com context.php** ✓
- **Problema:** Chatbot usava contexto básico ao invés do completo
- **Solução:** Agora usa o arquivo `context.php` com TODA a programação da rádio
- **Arquivo:** `groq_api.php` linha 39-48 e 195-198

### 4. **Logs de Debug Detalhados** ✓
- **Adicionado:** Logs completos em `chat_api.php` e `groq_api.php`
- **Localização:** `chatbot/logs/chatbot.log` e `chatbot/groq.log`

---

## 🚀 Páginas de Teste Disponíveis

### 1. **Teste Completo PHP** (RECOMENDADO)
```
http://localhost/chatbot/teste_completo.php
```
ou
```
http://seusite.com/chatbot/teste_completo.php
```

**O que testa:**
- ✓ Arquivos necessários
- ✓ Ambiente PHP e extensões
- ✓ Conexão real com API GROQ
- ✓ Sistema de logs
- ✓ Teste ao vivo com interface

**Vantagens:**
- Teste COMPLETO de todos os componentes
- Mostra quota da API GROQ
- Interface visual bonita
- Console de debug integrado

---

### 2. **Teste Rápido HTML**
```
http://localhost/chatbot/teste_rapido.html
```

**O que testa:**
- ✓ Comunicação frontend → backend
- ✓ Formato das mensagens
- ✓ Tratamento de erros
- ✓ Console de logs detalhado

**Vantagens:**
- Super rápido
- Não depende do WordPress
- Logs detalhados em tempo real
- Estatísticas de requisições

---

### 3. **Teste HTML Original**
```
http://localhost/chatbot/test_chatbot_html.html
```

**O que testa:**
- ✓ Carregamento do chatbot.js
- ✓ Carregamento do chatbot.css
- ✓ Inicialização do widget
- ✓ Verificação do DOM

---

## 🔍 Como Investigar Erros

### 1. Abra o Console do Navegador
**Chrome/Edge:** F12 → Console
**Firefox:** F12 → Console

### 2. Procure por mensagens do tipo:
```
[RadioChatbot INFO] Chatbot inicializado
[RadioChatbot INFO] API URL: http://localhost/chatbot/chat_api.php
```

### 3. Se aparecer erro, veja:
- ❌ `404 Not Found` → URL da API incorreta
- ❌ `Requisição inválida` → Problema no formato dos dados
- ❌ `Erro de conexão` → Servidor PHP não está rodando
- ❌ `CORS error` → Problema de permissões

---

## 📋 Checklist de Teste

### Teste Básico ✓
- [ ] Abrir `teste_completo.php`
- [ ] Verificar se todos os arquivos estão OK (✓ verde)
- [ ] Verificar se extensões PHP estão OK
- [ ] Verificar se conexão GROQ está funcionando
- [ ] Enviar mensagem de teste: "Olá!"
- [ ] Verificar se resposta aparece

### Teste de Contexto Completo ✓
- [ ] Perguntar: "Qual a frequência da rádio?"
  - Esperado: "105.5 MHz"
- [ ] Perguntar: "Que programa está no ar agora?"
  - Esperado: Nome do programa baseado no horário atual
- [ ] Perguntar: "Quais são os programas do Maurício Jacobi?"
  - Esperado: Lista completa de programas dele
- [ ] Perguntar: "Onde fica a rádio?"
  - Esperado: Rua Getúlio Vargas, 1425, Palmitos/SC

### Teste no Site WordPress ✓
- [ ] Abrir a página principal do site
- [ ] Verificar se ícone laranja aparece no canto inferior direito
- [ ] Clicar no ícone para abrir o chat
- [ ] Enviar mensagem
- [ ] Verificar resposta

---

## 🐛 Problemas Comuns e Soluções

### Problema 1: "Desculpe, requisição inválida"
**Causa:** Erro HTTP 400 da API GROQ
**Solução:** ✓ JÁ CORRIGIDO - formato do histórico agora compatível

### Problema 2: Chat não aparece no site
**Causa:** Plugin não ativado ou cache do WordPress
**Soluções:**
1. Verificar se plugin está ativo no admin do WordPress
2. Limpar cache do W3 Total Cache
3. Limpar cache do navegador (Ctrl + Shift + Delete)

### Problema 3: Mensagem "Erro de conexão"
**Causa:** chat_api.php não encontrado
**Soluções:**
1. Verificar se arquivo existe em `chatbot/chat_api.php`
2. Verificar permissões do arquivo (deve ter permissão de execução)
3. Testar diretamente acessando: `http://seusite.com/chatbot/chat_api.php`

### Problema 4: Contexto incompleto (não sabe a programação)
**Causa:** context.php não está sendo carregado
**Solução:** ✓ JÁ CORRIGIDO - agora carrega automaticamente

---

## 📊 Verificar Logs

### Logs do Backend (PHP)
```
chatbot/logs/chatbot.log   ← Conversas e erros gerais
chatbot/logs/groq.log       ← Logs da API GROQ
```

**Como ver:**
```bash
# Linux/Mac
tail -f chatbot/logs/groq.log

# Windows (PowerShell)
Get-Content chatbot/logs/groq.log -Wait
```

### Logs do Frontend (JavaScript)
Abra o **Console do navegador** (F12 → Console)

Procure por mensagens:
```
[RadioChatbot INFO] Chatbot inicializado
[RadioChatbot INFO] API URL: ...
[GroqAPI] Usando context.php completo
```

---

## ✅ Tudo Funcionando?

Se todos os testes passarem, você deve ver:

1. ✅ Chatbot aparece no site
2. ✅ Responde mensagens corretamente
3. ✅ Conhece toda a programação da rádio
4. ✅ Sabe a frequência, endereço, contatos
5. ✅ Responde sobre programas e apresentadores
6. ✅ Usa tom amigável e regional

---

## 🆘 Precisa de Ajuda?

Se algo não estiver funcionando:

1. **Primeiro:** Execute `teste_completo.php` e veja onde está o erro
2. **Segundo:** Verifique os logs em `chatbot/logs/`
3. **Terceiro:** Abra o console do navegador e procure erros
4. **Quarto:** Copie a mensagem de erro completa

---

## 🎯 Próximos Passos

Depois que tudo estiver funcionando:

1. **Desativar Debug:**
   - Abrir `chatbot.js` linha 544
   - Mudar `debug: true` para `debug: false`

2. **Limpar Cache:**
   - Limpar cache do WordPress (W3 Total Cache)
   - Limpar cache do navegador

3. **Testar em Produção:**
   - Testar em diferentes dispositivos
   - Testar em mobile
   - Pedir para alguém testar

4. **Monitorar:**
   - Verificar logs diariamente
   - Verificar quota da API GROQ
   - Ler feedback dos usuários

---

## 📝 Notas Importantes

- ⚠️ **API Key GROQ:** Está no código (linha 22 do groq_api.php). Em produção, mova para arquivo de config!
- ⚠️ **Limite diário:** GROQ free tem limite de ~14.400 requests/dia
- ⚠️ **Logs:** Podem crescer - limpar periodicamente
- ⚠️ **Cache:** W3 Total Cache pode cachear respostas antigas

---

**Última atualização:** 2025-01-17
**Versão do Chatbot:** 1.0.1 (com correções aplicadas)
