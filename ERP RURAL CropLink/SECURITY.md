# 🔒 Guia de Segurança - CropLink ERP

## ⚠️ Dados Sensíveis NÃO Incluídos

Este repositório está configurado para **NUNCA** incluir dados sensíveis no controle de versão.

### 🚫 Arquivos Protegidos (via .gitignore)

#### Credenciais e Configurações
- `.env` - Variáveis de ambiente
- `*.env` - Qualquer arquivo de ambiente
- `config/local.py` - Configurações locais
- `config/production.py` - Configurações de produção
- `local_settings.py` - Settings locais

#### Dados Sensíveis
- `secrets/` - Pasta de secrets
- `keys/` - Chaves de API
- `certs/` - Certificados SSL
- `cookies*.txt` - Cookies de sessão
- `headers.txt` - Headers HTTP

#### Bancos de Dados
- `*.db` - SQLite databases
- `*.sqlite` - SQLite databases
- `*.sqlite3` - SQLite databases

#### Arquivos Temporários e Gerados
- `attached_assets/` - Assets anexados
- `generated_images/` - Imagens geradas
- `logs/` - Arquivos de log

---

## ⚙️ Configuração Segura

### 1. Variáveis de Ambiente

**Copie o arquivo de exemplo:**
```bash
cp .env.example .env
```

**Configure as variáveis obrigatórias:**

```bash
# Gere uma SECRET_KEY forte
python -c "import secrets; print(secrets.token_hex(32))"

# Para Gmail, use senha de aplicativo:
# https://myaccount.google.com/apppasswords
```

### 2. Banco de Dados

**PostgreSQL (Produção):**
```bash
# Crie um usuário dedicado
CREATE USER croplink_user WITH PASSWORD 'senha-forte-aqui';

# Crie o banco
CREATE DATABASE croplink_db OWNER croplink_user;

# Grant permissões
GRANT ALL PRIVILEGES ON DATABASE croplink_db TO croplink_user;
```

**Nunca use:**
- Usuário `postgres` em produção
- Senhas fracas ou padrão
- Mesmo banco para dev e produção

### 3. Senhas de Administradores

O sistema requer senhas para contas administrativas especiais:

```env
ROOT_ADMIN_PASSWORD=senha-super-forte-com-min-16-caracteres
ALOIZIO_ADMIN_PASSWORD=outra-senha-forte-diferente
```

**Requisitos de senha:**
- Mínimo 12 caracteres
- Letras maiúsculas e minúsculas
- Números e símbolos
- Sem palavras do dicionário

---

## 🔐 Boas Práticas

### Para Desenvolvimento

1. **Nunca commite .env**
2. **Use bancos separados** (dev/test/prod)
3. **Senhas diferentes** para cada ambiente
4. **Rotação regular** de credentials

### Para Produção

1. **Use HTTPS sempre**
2. **Configure CORS** corretamente
3. **Habilite rate limiting**
4. **Monitore logs** de segurança
5. **Backups regulares** do banco
6. **Atualize dependências** regularmente

### Gerenciamento de Secrets

**Replit:**
- Use o painel "Secrets"
- Nunca coloque secrets em código

**Render/Heroku:**
- Use Environment Variables
- Configure via dashboard

**Docker:**
```bash
# Use secrets do Docker
docker secret create db_password ./db_password.txt
```

---

## 🚨 Checklist de Deploy

Antes de fazer deploy em produção:

- [ ] `.env` está no `.gitignore`
- [ ] SECRET_KEY é forte e única
- [ ] Senhas de admin são fortes
- [ ] DATABASE_URL usa SSL
- [ ] MAIL_PASSWORD é senha de app
- [ ] CORS está configurado corretamente
- [ ] Debug está `False`
- [ ] Logs não expõem dados sensíveis
- [ ] Backups estão configurados

---

## 🛡️ Recursos de Segurança do Sistema

### Implementados

✅ **Autenticação**
- Bcrypt para hash de senhas
- Login seguro com Flask-Login
- Sessões com timeout

✅ **Autorização**
- Sistema de níveis de acesso
- Isolamento de dados por usuário (multi-tenant)
- Aprovação de novos usuários

✅ **Proteções Web**
- CSRF protection (Flask-WTF)
- CORS configurável
- Secure cookies
- HTTP-only cookies

✅ **Banco de Dados**
- SQLAlchemy ORM (previne SQL injection)
- Prepared statements
- Pool de conexões seguro

✅ **Email**
- Validação de email
- Reset de senha seguro
- Templates sanitizados

### Recomendações Adicionais

Para produção em larga escala, considere:

- **Rate Limiting**: Flask-Limiter
- **WAF**: Cloudflare ou similar
- **Monitoring**: Sentry para erros
- **Backups**: Automatizados e encriptados
- **2FA**: Autenticação de dois fatores
- **Audit Logs**: Log de todas as ações sensíveis

---

## 📝 Reportando Vulnerabilidades

Se encontrar uma vulnerabilidade de segurança:

1. **NÃO** abra uma issue pública
2. Envie email para: security@croplink.com
3. Inclua detalhes técnicos e steps to reproduce
4. Aguarde resposta em até 48h

---

## 📚 Recursos

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Flask Security Best Practices](https://flask.palletsprojects.com/en/2.3.x/security/)
- [PostgreSQL Security](https://www.postgresql.org/docs/current/security.html)

---

> **Importante**: A segurança é responsabilidade de todos. Sempre revise código e configurações antes de deploy.
