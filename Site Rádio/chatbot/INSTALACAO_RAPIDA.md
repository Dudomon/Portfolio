# 🚀 Instalação Rápida - Chatbot Rádio Entre Rios

## ⏱️ 5 Minutos para o Chatbot Funcionar

### Passo 1: Obter API Key do Gemini (2 minutos)

1. Acesse: **https://makersuite.google.com/app/apikey**
2. Faça login com sua conta Google
3. Clique em **"Create API Key"**
4. Copie a chave (formato: `AIzaSy...`)

### Passo 2: Configurar a Chave (1 minuto)

1. Abra o arquivo: **`chatbot/config.php`**
2. Encontre a linha 10:
   ```php
   define('GEMINI_API_KEY', '');
   ```
3. Cole sua chave entre as aspas:
   ```php
   define('GEMINI_API_KEY', 'AIzaSy_SUA_CHAVE_AQUI');
   ```
4. Salve o arquivo

### Passo 3: Testar (1 minuto)

1. Abra no navegador: **`http://seu-site.com/chatbot/test.html`**
2. Clique no botão laranja flutuante
3. Digite: "Olá!"
4. Se receber resposta = ✅ **FUNCIONANDO!**

### Passo 4: Adicionar ao Site (1 minuto)

**WordPress:**

No arquivo `header.php` ou `footer.php` do seu tema, adicione antes do `</head>`:

```html
<link rel="stylesheet" href="/chatbot/chatbot.css">
<script src="/chatbot/chatbot.js"></script>
```

**HTML Estático:**

Adicione antes do `</body>`:

```html
<link rel="stylesheet" href="/chatbot/chatbot.css">
<script src="/chatbot/chatbot.js"></script>
```

---

## ✅ Pronto!

O chatbot já está funcionando no seu site!

### 🎯 Próximos Passos (Opcional)

- Personalize as cores em `chatbot/chatbot.css`
- Ajuste o contexto em `chatbot/context.php`
- Configure programação e horários
- Adicione mais FAQs

---

## ❓ Problemas?

### Chatbot não aparece
- Verifique se os arquivos CSS/JS foram incluídos
- Abra o Console (F12) e veja se há erros

### Erro "API Key não configurada"
- Verifique se salvou o arquivo `config.php`
- Confirme que a chave está entre aspas simples: `'...'`
- Não deixe espaços extras

### "Erro ao processar mensagem"
- Verifique se PHP cURL está habilitado
- Confirme que a API Key é válida
- Veja os logs em `chatbot/logs/chatbot.log`

---

## 📖 Documentação Completa

Consulte **`README.md`** para:
- Personalização avançada
- Integração com APIs
- Monitoramento e analytics
- Troubleshooting detalhado

---

## 💰 Custos

**GRÁTIS** até 1500 conversas/dia!

Acima disso: ~R$ 0,40 por 1000 mensagens (muito barato!)

---

**Desenvolvido com ❤️ para Rádio Entre Rios FM 99.1 MHz**
