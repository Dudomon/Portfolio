# 🚀 Guia de Instalação no WordPress - Chatbot Chatinho

## 📦 Onde Colocar os Arquivos

### ⚠️ IMPORTANTE: Pasta na RAIZ do site, NÃO no wp-content!

```
/public_html/               ← Raiz do site (ou /htdocs/, /www/)
├── wp-admin/
├── wp-content/
├── wp-includes/
├── index.php
├── wp-config.php
└── chatbot/                ← COLOCAR AQUI!
    ├── config.php
    ├── context.php
    ├── chat_api.php
    ├── chatbot.js
    ├── chatbot.css
    └── logs/
```

---

## 🎯 Método 1: Plugin WordPress (RECOMENDADO)

### Vantagens:
- ✅ Mais fácil de instalar
- ✅ Pode ativar/desativar pelo painel
- ✅ Página de configurações no admin
- ✅ Não mexe no código do tema

### Passo a Passo:

**1. Upload via FTP:**

a) Conecte no FTP do site (FileZilla, por exemplo)
b) Acesse a pasta raiz: `/public_html/` ou `/home/usuario/public_html/`
c) Faça upload da pasta `chatbot/` inteira para lá

**2. Instalar o Plugin:**

a) Copie o arquivo `radio-chatbot-plugin.php` para:
   ```
   /wp-content/plugins/radio-chatbot-plugin.php
   ```

b) No WordPress, vá em: **Plugins > Plugins Instalados**

c) Ative o plugin **"Rádio Entre Rios - Chatbot Chatinho"**

**3. Pronto!** 🎉

O chatbot já está funcionando em todas as páginas!

### Ver Configurações:

No painel do WordPress, vá em:
```
Configurações > Chatbot
```

---

## 🎯 Método 2: Via Functions.php

### Vantagens:
- ✅ Ainda mais simples (sem plugin)
- ✅ Integração direta no tema

### Desvantagens:
- ⚠️ Se trocar de tema, precisa adicionar novamente

### Passo a Passo:

**1. Upload via FTP:**

Igual ao Método 1: Faça upload da pasta `chatbot/` para a raiz do site.

**2. Editar Functions.php:**

a) No WordPress, vá em: **Aparência > Editor de Arquivos de Temas**

b) Abra o arquivo `functions.php`

c) **No final do arquivo**, adicione este código:

```php
/**
 * Adiciona o Chatbot Chatinho no site
 */
function radioentrerios_add_chatbot() {
    wp_enqueue_style(
        'chatbot-style',
        get_site_url() . '/chatbot/chatbot.css',
        array(),
        '1.0.0'
    );

    wp_enqueue_script(
        'chatbot-script',
        get_site_url() . '/chatbot/chatbot.js',
        array(),
        '1.0.0',
        true
    );
}
add_action('wp_enqueue_scripts', 'radioentrerios_add_chatbot');
```

d) Clique em **"Atualizar Arquivo"**

**3. Pronto!** 🎉

---

## 🎯 Método 3: HTML Direto (Simples)

Se você tem acesso ao header.php ou footer.php do tema:

**1. Upload via FTP:**

Faça upload da pasta `chatbot/` para a raiz do site.

**2. Editar Header ou Footer:**

a) No WordPress, vá em: **Aparência > Editor de Arquivos de Temas**

b) Abra `header.php` ou `footer.php`

c) Antes do `</head>` (no header) ou antes do `</body>` (no footer), adicione:

```html
<!-- Chatbot Chatinho -->
<link rel="stylesheet" href="<?php echo get_site_url(); ?>/chatbot/chatbot.css">
<script src="<?php echo get_site_url(); ?>/chatbot/chatbot.js"></script>
```

d) Salvar

**3. Pronto!** 🎉

---

## 📂 Checklist de Instalação

Antes de testar, verifique:

- [ ] Pasta `chatbot/` está na **raiz** do site (não em wp-content)
- [ ] Arquivo `config.php` tem a **API Key do Gemini**
- [ ] Permissões da pasta `logs/` (chmod 755)
- [ ] Código adicionado no WordPress (plugin, functions.php ou header/footer)

---

## 🧪 Testar Instalação

### 1. Teste Técnico:

Acesse no navegador:
```
https://www.radioentrerios.com.br/chatbot/test.html
```

Se aparecer a página de testes = ✅ Arquivos no lugar correto!

### 2. Teste no Site:

Acesse qualquer página do site:
```
https://www.radioentrerios.com.br
```

Deve aparecer um **botão laranja flutuante** no canto inferior direito.

### 3. Teste de Conversa:

Clique no botão e teste perguntas:
- "Qual a frequência da rádio?"
- "Qual o horário do Bola em Jogo?"
- "Quem apresenta o Coração Sertanejo?"

---

## 🔧 Troubleshooting

### Problema: Botão não aparece

**Solução:**
1. Abra o Console do navegador (F12)
2. Vá na aba "Console"
3. Veja se há erros de JavaScript
4. Verifique se os arquivos CSS/JS foram carregados (aba "Network")

**Possíveis causas:**
- Caminho errado dos arquivos
- Arquivos não foram enviados via FTP
- Cache do navegador (Ctrl+F5 para limpar)

### Problema: Chatbot não responde

**Solução:**
1. Verifique se a API Key está configurada em `config.php`
2. Teste a API: `https://seu-site.com/chatbot/diagnostico.php`
3. Veja os logs: `/chatbot/logs/chatbot.log`

### Problema: Erro 404 ao clicar

**Solução:**
- Certifique-se de que a pasta está na raiz, não em wp-content
- Verifique permissões dos arquivos (644) e pastas (755)

---

## 🎨 Personalização (Opcional)

### Mudar Cores:

Edite `chatbot/chatbot.css`:
```css
:root {
    --chatbot-primary: #FF6B00;  /* Cor principal */
}
```

### Mudar Posição:

Edite `chatbot/chatbot.js`:
```javascript
position: 'bottom-right'  // ou 'bottom-left'
```

### Atualizar Programação:

Edite `chatbot/context.php` e adicione/remova programas conforme necessário.

---

## 📊 Monitoramento

### Ver Conversas:

Via FTP, baixe o arquivo:
```
/chatbot/logs/chatbot.log
```

### Estatísticas:

No futuro, poderemos adicionar um dashboard de analytics no painel do WordPress.

---

## 💰 Custos

- **Gemini API:** Grátis até 1500 conversas/dia
- **Hospedagem:** Não requer recursos extras
- **Total:** R$ 0,00 (tier gratuito)

---

## 📞 Suporte

Se precisar de ajuda:
1. Verifique este guia
2. Teste com `diagnostico.php`
3. Veja os logs em `/chatbot/logs/`

---

## ✅ Resumo Rápido

1. **Upload:** Pasta `chatbot/` na raiz do site (via FTP)
2. **Integração:** Instalar plugin OU adicionar código no functions.php
3. **Testar:** Acessar o site e ver o botão laranja
4. **Conversar:** "Oi Chatinho!"

---

**Versão:** 1.0
**Data:** Janeiro 2025
**Desenvolvido para:** Rádio Entre Rios FM 105.5 MHz
