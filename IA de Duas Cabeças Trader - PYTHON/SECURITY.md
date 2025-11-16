# 🔒 Guia de Segurança - IA Trading System

## ⚠️ Arquivos Sensíveis NÃO Incluídos

Por questões de segurança, os seguintes arquivos **NÃO** estão incluídos neste repositório:

### 🔐 Credenciais e API Keys
- `reset_bin_passwords.py` - Contém API keys do JSONBin
- `gerenciar_usuarios_online.py` - Sistema de gerenciamento de usuários
- `online_system_real.py` - Configurações de autenticação online
- `online_login_ander.py` - Sistema de login
- `usuarios.db` - Banco de dados de usuários

### 🤖 Modelos Treinados
- Todos os arquivos `.zip` (modelos podem ter 100MB+)
- Pasta `Modelos para testar/`
- Pasta `Modelo daytrade/`
- Checkpoints de treinamento

### 📊 Dados de Trading
- Arquivos `.db` e `.sqlite`
- Logs de trading com informações sensíveis
- Histórico de operações

---

## 🛠️ Como Configurar o Sistema

### 1. Configuração de Credenciais

Copie o arquivo de exemplo e configure suas credenciais:

```bash
cp "Modelo PPO Trader/online_system_real.py.example" "Modelo PPO Trader/online_system_real.py"
```

Edite o arquivo e preencha:
- `ONLINE_API_KEY` - Sua chave de API do JSONBin (https://jsonbin.io/)
- `USERS_BIN_ID` - ID do seu Bin de usuários

### 2. Estrutura de Dados

O sistema espera a seguinte estrutura de usuários:

```json
{
  "users": {
    "username": {
      "password_hash": "hash_sha256_da_senha",
      "expires_at": "2024-12-31",
      "active": true
    }
  }
}
```

### 3. Variáveis de Ambiente (Recomendado)

Para maior segurança, use variáveis de ambiente:

```python
import os

ONLINE_API_KEY = os.getenv('JSONBIN_API_KEY')
USERS_BIN_ID = os.getenv('JSONBIN_USERS_ID')
```

---

## 🚫 O que NUNCA Commitar

- ✗ API Keys e tokens
- ✗ Senhas (mesmo hasheadas)
- ✗ Arquivos `.db` ou `.sqlite`
- ✗ Modelos treinados (use Git LFS ou armazenamento separado)
- ✗ Logs com dados de operações reais
- ✗ Credenciais de MetaTrader5

---

## ✅ Boas Práticas

1. **Use `.env` para credenciais locais**
2. **Gere API keys separadas para dev/prod**
3. **Rotacione suas credenciais periodicamente**
4. **Use Git LFS para modelos grandes** (se necessário compartilhar)
5. **Mantenha backups seguros dos modelos treinados**

---

## 📝 Licença e Uso

Este código é disponibilizado para fins educacionais e de portfólio.

**AVISO:** Este sistema opera com trading real. Use por sua conta e risco.
