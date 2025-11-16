# 🎬 SCREENCAST PARA APROVAÇÃO META API

## 📋 **PREPARAÇÃO PRE-GRAVAÇÃO**

### **1. CONFIGURAR AMBIENTE EM INGLÊS**
```
- Windows: Settings > Language > English (US)
- Browser: Configure para inglês
- Sistema: Alterar região para EUA temporariamente
```

### **2. LINKS ESSENCIAIS DO SISTEMA**

**🔗 URLs PRINCIPAIS:**
- **Site da Rádio:** https://radioentrerios.com.br
- **Admin WordPress:** https://radioentrerios.com.br/wp-admin  
- **Sistema de Notícias:** https://radioentrerios.com.br/wp-content/noticias/
- **API de Notícias:** https://radioentrerios.com.br/wp-content/noticias/get_noticias.php

**📡 ENDPOINTS ESPECÍFICOS PARA DEMONSTRAÇÃO:**

#### **1. PAGE PUBLIC CONTENT ACCESS**
```
DEMONSTRAR LENDO POSTS PÚBLICOS:
https://radioentrerios.com.br/wp-content/noticias/facebook_posts_demo.php

Deve mostrar:
- Posts coletados de páginas públicas
- Processamento de conteúdo
- Integração com sistema de notícias
```

#### **2. INSTAGRAM_BUSINESS_BASIC**
```
MOSTRAR DADOS DA CONTA BUSINESS:
https://radioentrerios.com.br/wp-content/noticias/instagram_account_demo.php

Deve retornar:
- Nome da conta: @radioentrerios
- ID do negócio  
- Followers count
- Profile info básico
```

#### **3. PAGES_SHOW_LIST**
```
LISTAR PÁGINAS DISPONÍVEIS:
https://radioentrerios.com.br/wp-content/noticias/facebook_pages_demo.php

Deve mostrar:
- Rádio Entre Rios Facebook Page
- Outras páginas conectadas
- IDs e permissões de cada página
```

#### **4. INSTAGRAM_BUSINESS_MANAGE_MESSAGES**
```
DASHBOARD DE MENSAGENS:
https://radioentrerios.com.br/wp-content/noticias/instagram_messages_demo.php

Deve mostrar:
- Mensagens recebidas do público
- Respostas automáticas sobre notícias
- Status de conversas ativas
```

#### **5. ADS_READ**  
```
DASHBOARD DE MÉTRICAS:
https://radioentrerios.com.br/wp-content/noticias/ads_analytics_demo.php

Deve mostrar:
- Alcance dos posts de notícias
- Engagement metrics
- Performance de promoção de conteúdo
```

---

## 🎥 **ROTEIRO DETALHADO (7 MINUTOS)**

### **INTRO (30 segundos)**
```
🎬 CENA 1: APRESENTAÇÃO
- Abrir site: https://radioentrerios.com.br
- Mostrar widget de notícias funcionando
- Focar em notícias municipais prioritárias

📢 NARRAÇÃO EM INGLÊS:
"This is Radio Entre Rios News Automation System. We serve a local Brazilian 
community by collecting municipal news and distributing across social media 
channels automatically, prioritizing local content over national news."

📝 TEXTO NA TELA:
"Radio Entre Rios - Local News Automation"
"Serving Brazilian Community Since 2015"
```

### **LOGIN FLOW (60 segundos)**
```
🎬 CENA 2: AUTENTICAÇÃO META
- Mostrar página de configuração do sistema
- Demonstrar conexão com Meta APIs
- Processo de OAuth completo

📢 NARRAÇÃO:
"System administrators authenticate through Meta's OAuth system to connect 
business Facebook and Instagram accounts. This enables automated content 
distribution while maintaining security standards."

📝 TEXTO NA TELA:
"Step 1: Meta API Authentication"
"Secure OAuth Integration"
```

### **PERMISSION USAGE - PARTE 1 (90 segundos)**
```
🎬 CENA 3: PAGE PUBLIC CONTENT ACCESS + PAGES_SHOW_LIST
- Abrir: facebook_posts_demo.php
- Mostrar coleta de posts públicos
- Abrir: facebook_pages_demo.php  
- Mostrar lista de páginas conectadas

📢 NARRAÇÃO:
"Page Public Content Access permission allows us to read public posts from 
local news sources and government pages. Pages Show List displays all 
connected Facebook pages where we can distribute content."

📝 TEXTO NA TELA:
"Reading Public Posts from Municipal Sources"
"Managing Connected Facebook Pages"
```

### **PERMISSION USAGE - PARTE 2 (90 segundos)**
```
🎬 CENA 4: INSTAGRAM_BUSINESS_BASIC + MANAGE_MESSAGES
- Abrir: instagram_account_demo.php
- Mostrar dados da conta business
- Abrir: instagram_messages_demo.php
- Mostrar mensagens automáticas

📢 NARRAÇÃO:
"Instagram Business Basic provides essential account information for content 
publishing. Instagram Business Manage Messages enables automated responses 
to community inquiries about local news and events."

📝 TEXTO NA TELA:
"Instagram Business Account Integration"
"Automated Community Message Management"
```

### **PERMISSION USAGE - PARTE 3 (60 segundos)**
```
🎬 CENA 5: ADS_READ
- Abrir: ads_analytics_demo.php
- Mostrar métricas de alcance
- Demonstrar otimização baseada em dados

📢 NARRAÇÃO:
"Ads Read permission provides analytics data to optimize content distribution. 
We analyze reach, engagement, and community interaction to improve local 
news delivery effectiveness."

📝 TEXTO NA TELA:
"Analytics-Driven Content Optimization"
"Maximizing Community Reach"
```

### **DATA COLLECTION (90 segundos)**
```
🎬 CENA 6: COLETA AUTOMATIZADA
- Abrir: news_system_php/news_collector.php
- Executar coleta em tempo real
- Mostrar priorização de conteúdo municipal

📢 NARRAÇÃO:
"The system automatically collects news from municipal websites, regional 
RSS feeds, and social media sources. Local municipal content receives 
highest priority in our algorithm."

📝 TEXTO NA TELA:
"Automated News Collection Process"
"Municipal Priority Algorithm Active"
```

### **FINAL RESULT (60 segundos)**
```
🎬 CENA 7: RESULTADO INTEGRADO
- Mostrar site com notícias atualizadas
- Mostrar Facebook page com posts automáticos
- Mostrar Instagram com conteúdo distribuído
- Demonstrar engajamento da comunidade

📢 NARRAÇÃO:
"Final result: Complete automated news ecosystem serving local community 
with prioritized municipal content, distributed across digital channels, 
with analytics-driven optimization for maximum community engagement."

📝 TEXTO NA TELA:
"Complete Local News Automation"
"Serving 15,000+ Community Members"
```

---

## 🛠 **FERRAMENTAS DE GRAVAÇÃO**

### **RECOMENDADO: OBS Studio (Gratuito)**
- **Download:** https://obsproject.com/
- **Configurações:**
  ```
  Resolution: 1920x1080 (Full HD)
  FPS: 30
  Audio: 44.1kHz, Stereo
  Format: MP4 (H.264)
  Bitrate: 2500-5000 kbps
  ```

### **ALTERNATIVAS:**
- **Camtasia** (pago): Interface mais simples
- **ScreencastO-Matic**: Versão online
- **Loom**: Gravação rápida na nuvem

---

## 📝 **SCRIPT COMPLETO EM INGLÊS**

```
INTRODUCTION:
"Welcome to the Radio Entre Rios News Automation System demonstration. 
This comprehensive solution serves a local Brazilian community by 
collecting, processing, and automatically distributing municipal news 
with priority over national content."

META API INTEGRATION:
"Our system integrates with Meta's Graph API through secure OAuth 
authentication, requesting specific permissions for legitimate news 
automation and community engagement purposes."

PERMISSION DEMONSTRATIONS:
"Page Public Content Access enables reading from municipal government 
pages and local news sources. Instagram Business Basic provides 
essential account management capabilities. Pages Show List allows 
content distribution across connected Facebook pages. Instagram 
Business Manage Messages handles automated community responses. 
Ads Read provides analytics for content optimization."

AUTOMATED WORKFLOW:
"The system continuously monitors local news sources, applies 
municipal priority algorithms, and distributes content across 
digital channels while maintaining proper attribution and 
community engagement tracking."

COMMUNITY IMPACT:
"This automation serves over 15,000 community members with timely, 
relevant local information, strengthening civic engagement and 
community awareness through modern digital infrastructure."
```

---

## ⚠️ **CHECKLIST PRÉ-UPLOAD**

### **QUALIDADE TÉCNICA:**
- [ ] Resolução mínima 1080p
- [ ] Áudio claro e sem ruído
- [ ] Legendas em inglês precisas
- [ ] Duração: 5-8 minutos
- [ ] Formato MP4 (H.264)

### **CONTEÚDO OBRIGATÓRIO:**
- [ ] Login flow completo da Meta
- [ ] Uso específico de cada permissão solicitada
- [ ] Resultado final visível e funcional
- [ ] Interface do usuário em inglês
- [ ] Explicação clara de cada funcionalidade

### **COMPLIANCE:**
- [ ] Sem informações sensíveis visíveis
- [ ] Dados pessoais censurados/mockados
- [ ] URLs públicas funcionais
- [ ] Demonstração de casos de uso legítimos

---

## 🚀 **PRÓXIMOS PASSOS**

1. **CRIAR ENDPOINTS DEMO** (se necessário)
2. **CONFIGURAR AMBIENTE EM INGLÊS**
3. **GRAVAR SCREENCAST SEGUINDO ROTEIRO**
4. **REVISAR E EDITAR** 
5. **UPLOAD NO META APP REVIEW**

---

*Documentação criada em: Agosto 2025*  
*Para aprovação das APIs: Page Public Content Access, ads_read, pages_show_list, instagram_business_manage_messages, instagram_business_basic*