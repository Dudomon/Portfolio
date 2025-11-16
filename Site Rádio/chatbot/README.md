# 🤖 Chatbot Rádio Entre Rios

Sistema completo de chatbot inteligente com integração do Google Gemini para o site da Rádio Entre Rios FM 99.1 MHz.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Integração com o Site](#integração-com-o-site)
- [Personalização](#personalização)
- [Testes](#testes)
- [Estrutura de Arquivos](#estrutura-de-arquivos)
- [FAQ](#faq)
- [Suporte](#suporte)

---

## 🎯 Visão Geral

O **Assis** é o assistente virtual da Rádio Entre Rios, desenvolvido usando a API do Google Gemini. Ele oferece atendimento 24/7 aos ouvintes, respondendo perguntas sobre:

- 📻 Programação da rádio
- 📞 Informações de contato
- 🎵 Pedidos musicais
- 📰 Notícias locais
- 🛠️ Suporte técnico (player, streaming)
- ℹ️ Informações sobre Palmitos e região

---

## ✨ Funcionalidades

### Interface
- ✅ Widget flutuante responsivo
- ✅ Design moderno com gradientes (laranja/ciano)
- ✅ Animações suaves
- ✅ Modo escuro automático
- ✅ Notificações quando o chat está fechado
- ✅ Histórico de conversas (salvo no navegador)

### Backend
- ✅ Integração com Google Gemini 1.5 Flash
- ✅ Context injection (conhecimento sobre a rádio)
- ✅ Rate limiting (proteção contra spam)
- ✅ Sanitização de entrada (proteção XSS)
- ✅ Logs de conversas
- ✅ Tratamento de erros

### Segurança
- ✅ Validação de entrada
- ✅ CORS configurado
- ✅ Rate limiting por IP
- ✅ Escape de HTML
- ✅ Timeout de requisições

---

## 🚀 Instalação

### Pré-requisitos

1. **Servidor PHP** (versão 7.4 ou superior)
2. **Extensão cURL habilitada** no PHP
3. **Chave da API Google Gemini** (gratuita)

### Passo 1: Obter Chave da API Gemini

1. Acesse: https://makersuite.google.com/app/apikey
2. Faça login com sua conta Google
3. Clique em "Create API Key"
4. Copie a chave gerada (formato: `AIza...`)

### Passo 2: Configurar a Chave

Abra o arquivo `chatbot/config.php` e adicione sua chave:

```php
define('GEMINI_API_KEY', 'SUA_CHAVE_AQUI');
```

### Passo 3: Verificar Permissões

Certifique-se de que o diretório `chatbot/logs/` tem permissões de escrita:

```bash
chmod 755 chatbot/logs/
```

### Passo 4: Testar

Abra no navegador:
```
http://seu-dominio.com/chatbot/test.html
```

---

## ⚙️ Configuração

### Arquivo: `chatbot/config.php`

```php
// API Gemini
define('GEMINI_API_KEY', '');  // Sua chave aqui

// Personalização
define('CHATBOT_NAME', 'Assis - Rádio Entre Rios');
define('CHATBOT_WELCOME_MESSAGE', 'Olá! Como posso ajudar?');

// Limites
define('MAX_REQUESTS_PER_IP', 30);  // Requests por hora
define('MAX_MESSAGE_LENGTH', 500);  // Tamanho máximo da mensagem

// Logs
define('DEBUG_MODE', true);  // Mude para false em produção
define('LOG_CONVERSATIONS', true);
```

### Arquivo: `chatbot/context.php`

Aqui você pode editar o conhecimento do chatbot sobre a rádio:

- Programação
- Contatos
- FAQs
- Tom de voz
- Respostas padrão

---

## 🌐 Integração com o Site

### Método 1: WordPress (Recomendado)

Adicione este código no arquivo `header.php` ou `footer.php` do tema:

```html
<!-- Chatbot da Rádio Entre Rios -->
<link rel="stylesheet" href="/chatbot/chatbot.css">
<script src="/chatbot/chatbot.js"></script>
```

### Método 2: HTML Estático

Adicione antes do `</body>`:

```html
<!-- Chatbot -->
<link rel="stylesheet" href="/chatbot/chatbot.css">
<script src="/chatbot/chatbot.js"></script>
```

### Método 3: Plugin WordPress

Crie um plugin customizado:

1. Crie o arquivo: `wp-content/plugins/radio-chatbot.php`

```php
<?php
/*
Plugin Name: Rádio Entre Rios Chatbot
Description: Chatbot inteligente com Gemini
Version: 1.0
*/

function radio_chatbot_scripts() {
    wp_enqueue_style('radio-chatbot-css', '/chatbot/chatbot.css');
    wp_enqueue_script('radio-chatbot-js', '/chatbot/chatbot.js', array(), '1.0', true);
}
add_action('wp_enqueue_scripts', 'radio_chatbot_scripts');
?>
```

2. Ative o plugin no painel do WordPress

---

## 🎨 Personalização

### Cores

Edite `chatbot/chatbot.css`:

```css
:root {
    --chatbot-primary: #FF6B00;  /* Cor principal (laranja) */
    --chatbot-secondary: #00BFFF; /* Cor secundária (ciano) */
}
```

### Posição

Edite `chatbot/chatbot.js`:

```javascript
window.radioChatbot = new RadioChatbot({
    position: 'bottom-right',  // ou 'bottom-left'
});
```

### Mensagem de Boas-vindas

Edite `chatbot/config.php`:

```php
define('CHATBOT_WELCOME_MESSAGE', 'Sua mensagem personalizada aqui');
```

### Contexto e Conhecimento

Edite `chatbot/context.php` para:
- Atualizar programação
- Adicionar novos contatos
- Modificar FAQs
- Ajustar tom de voz
- Adicionar informações sazonais

---

## 🧪 Testes

### Teste Local

1. Abra: `http://localhost/chatbot/test.html`
2. Clique no botão flutuante laranja
3. Digite uma mensagem de teste

### Testes Recomendados

```
✅ "Qual a frequência da rádio?"
✅ "Como ouvir online?"
✅ "Qual o telefone para contato?"
✅ "Onde fica a rádio?"
✅ "Como fazer pedido musical?"
✅ "Quais são as notícias de hoje?"
✅ "O player não está funcionando"
✅ "Quero falar com um atendente"
```

### Verificar Logs

```bash
tail -f chatbot/logs/chatbot.log
```

---

## 📁 Estrutura de Arquivos

```
chatbot/
├── config.php           # Configurações principais
├── context.php          # Base de conhecimento da rádio
├── chat_api.php         # Backend que integra com Gemini
├── chatbot.js           # Interface do widget
├── chatbot.css          # Estilos do chatbot
├── test.html            # Página de testes
├── README.md            # Esta documentação
└── logs/
    ├── chatbot.log      # Log de conversas
    └── ratelimit_*.txt  # Cache de rate limiting
```

---

## 📊 Monitoramento

### Analytics

O chatbot registra:
- Número de conversas
- Perguntas mais frequentes
- Horários de pico
- Erros da API

### Visualizar Estatísticas

```bash
# Conversas de hoje
grep "$(date +%Y-%m-%d)" chatbot/logs/chatbot.log | wc -l

# Perguntas mais comuns
grep "User:" chatbot/logs/chatbot.log | sort | uniq -c | sort -rn | head -10

# Erros recentes
grep "ERROR" chatbot/logs/chatbot.log | tail -20
```

---

## 🔧 Troubleshooting

### Problema: "API Key não configurada"

**Solução:**
1. Verifique se adicionou a chave em `config.php`
2. Certifique-se de que não há espaços extras
3. Verifique se a chave está entre aspas: `'AIza...'`

### Problema: Chatbot não aparece

**Solução:**
1. Abra o Console do navegador (F12)
2. Verifique se há erros de JavaScript
3. Confirme que os arquivos CSS/JS foram carregados
4. Verifique o caminho dos arquivos

### Problema: "Erro ao processar mensagem"

**Solução:**
1. Verifique os logs: `chatbot/logs/chatbot.log`
2. Confirme que a extensão cURL está habilitada
3. Teste a conexão com a API Gemini manualmente
4. Verifique se a API Key é válida

### Problema: Rate Limit atingido

**Solução:**
1. Aguarde 1 hora (limite reseta automaticamente)
2. Ou limpe o cache: `rm chatbot/logs/ratelimit_*.txt`
3. Ajuste o limite em `config.php` se necessário

---

## 💡 Dicas de Otimização

### Performance

1. **Minificar arquivos em produção:**
```bash
# Minificar chatbot.js
uglifyjs chatbot.js -o chatbot.min.js -c -m

# Minificar chatbot.css
cleancss chatbot.css -o chatbot.min.css
```

2. **Habilitar cache no servidor:**
```apache
# .htaccess
<FilesMatch "\.(js|css)$">
    Header set Cache-Control "max-age=86400, public"
</FilesMatch>
```

### Custos

- **Gemini 1.5 Flash:** 1500 requests/dia grátis
- **Após limite:** ~$0.075 por 1000 requests
- **Estimativa:** Para 500 conversas/dia = R$ 0 (tier gratuito)

---

## 🔄 Atualizações Futuras

Recursos planejados:

- [ ] Integração com API de notícias (`get_noticias.php`)
- [ ] Integração com API de podcasts (`get_podcasts.php`)
- [ ] Busca de música tocando agora
- [ ] Histórico de conversas no servidor
- [ ] Dashboard de analytics
- [ ] Respostas com áudio (TTS)
- [ ] Integração com WhatsApp Business
- [ ] Multi-idioma (PT/ES/EN)

---

## 📞 Suporte

### Problemas Técnicos

1. Verifique a documentação acima
2. Consulte os logs em `chatbot/logs/chatbot.log`
3. Teste com `test.html` primeiro

### Contato

- **Rádio:** (49) 3647-0292
- **WhatsApp:** (49) 99116-9292
- **Email:** contato@radioentrerios.com.br

---

## 📄 Licença

© 2025 Rádio Entre Rios FM - Todos os direitos reservados.

Este chatbot foi desenvolvido exclusivamente para uso da Rádio Entre Rios.

---

## 🙏 Créditos

- **IA:** Google Gemini 1.5 Flash
- **Desenvolvedor:** Claude (Anthropic)
- **Design:** Baseado nas cores da Rádio Entre Rios

---

**Versão:** 1.0
**Última atualização:** Janeiro 2025
**Status:** ✅ Produção
