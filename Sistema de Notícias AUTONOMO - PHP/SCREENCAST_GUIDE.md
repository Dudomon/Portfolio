# 🎥 Guia para Screencast - Meta App Review

## 📋 O que a Meta quer ver:

1. **Seu app enviando uma mensagem para um usuário Instagram**
2. **O Instagram (web ou mobile) recebendo e exibindo a mensagem**

## 🎯 Cenário de Demonstração

### Preparação (antes de gravar):

1. **Abra duas telas/dispositivos:**
   - 🖥️ **Tela 1**: Navegador com `test_messages.php` 
   - 📱 **Tela 2**: Instagram mobile OU Instagram web

2. **Contas necessárias:**
   - Conta Instagram Business (@entreriosfm105.5)
   - Conta Instagram pessoal (para receber a mensagem)

3. **Obtenha o Instagram User ID do destinatário:**
   - Use: https://www.instagram.com/web/search/topsearch/?query=USERNAME
   - Ou ferramenta: https://commentpicker.com/instagram-user-id.php

### 🎬 Roteiro do Screencast:

#### Parte 1: Configuração (30 segundos)
```
"Este é nosso sistema de mensagens Instagram da Rádio Entre Rios.
Vou demonstrar o envio de uma mensagem do nosso app para um usuário 
e mostrar o recebimento no Instagram."
```

1. Mostre a tela `test_messages.php` aberta
2. Mostre que está autenticado (✅ Autenticado como: Rádio Entre Rios)
3. Mostre o Instagram aberto na conta de destino

#### Parte 2: Envio da Mensagem (1 minuto)
```
"Agora vou enviar uma mensagem com as informações da nossa rádio:"
```

1. **Preencha o formulário:**
   - ID do destinatário: `[SEU_INSTAGRAM_USER_ID]`
   - Clique em "📻 Enviar Info da Rádio"

2. **Mostre o resultado:**
   - ✅ Mensagem enviada com sucesso
   - Resposta da API com detalhes

#### Parte 3: Recebimento no Instagram (30 segundos)
```
"Agora vou mostrar a mensagem chegando no Instagram:"
```

1. **Mude para o Instagram**
2. **Mostre a mensagem chegando:**
   - Vá para Direct Messages (DMs)
   - Mostre a mensagem da @entreriosfm105.5
   - Leia o conteúdo da mensagem

#### Parte 4: Interação (30 segundos)
```
"Para demonstrar a interação completa, vou responder à mensagem:"
```

1. **No Instagram, responda algo como:**
   - "Obrigado pelas informações!"
   - Ou "Gostaria de saber mais sobre a programação"

2. **Volte para o app, mostre que pode receber/processar respostas**

## 📱 URLs e Comandos

### Para acessar o sistema:
```
Local: D:\Widget noticias\instagram\test_messages.php
Servidor: https://radioentrerios.com.br/wp-content/noticias/instagram/test_messages.php
```

### IDs de exemplo para teste:
```
- Use seu próprio Instagram User ID
- Ou ID de conta teste criada especificamente
```

## ⚠️ Pontos Importantes:

1. **Mostre URL completa** do seu sistema na barra do navegador
2. **Mantenha gravação contínua** - sem cortes
3. **Narração clara** explicando cada passo
4. **Tempo total**: 2-3 minutos máximo
5. **Qualidade**: HD (1080p) mínimo

## 🔧 Se der erro:

### "Instagram User ID não encontrado":
- Use ferramenta online para obter ID correto
- Teste com sua própria conta primeiro

### "Permissões insuficientes":
- Refaça autenticação em `auth.php`  
- Verifique se estas permissões foram aprovadas:
  - ads_read
  - pages_show_list  
  - instagram_business_manage_messages
  - instagram_business_basic
  - Page Public Content Access

### "Token expirado":
- Refaça autenticação completa
- Gere novo token de longa duração

## 📄 Mensagem exemplo que será enviada:

```
🎵 Rádio Entre Rios 105.5 FM

📻 Escute ao vivo: radioentrerios.com.br
📱 Baixe nosso app na Play Store
📰 Últimas notícias disponíveis 24h

Entre em contato conosco:
📞 (49) 3344-3600
📧 contato@radioentrerios.com.br
```

## 🎯 Resultado esperado:

Ao final do screencast, a Meta deve ver claramente:
- ✅ App enviando mensagem via Instagram API
- ✅ Mensagem chegando no Instagram do destinatário
- ✅ Fluxo completo de comunicação funcionando
- ✅ Uso legítimo da API para comunicação com ouvintes

---

**Pronto para gravar! 🚀**