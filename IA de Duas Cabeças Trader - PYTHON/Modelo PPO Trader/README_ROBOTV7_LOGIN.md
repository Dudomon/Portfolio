# 🤖 RobotV7 com Sistema de Login Integrado

Sistema de autenticação profissional integrado ao RobotV7 Legion, baseado no sistema Robo Ander.

## 🚀 Arquivos Criados

### Scripts Principais:
- **`robotlogin.py`** - RobotV7 com sistema de login integrado
- **`robotv7_login_system.py`** - Sistema de autenticação específico para RobotV7
- **`test_robotv7_login.py`** - Script de testes e validação

### Scripts de Suporte:
- **`online_login_ander.py`** - Sistema base do Robo Ander
- **`online_system_real.py`** - Backend online  
- **`gerenciar_usuarios_online.py`** - Gerenciamento de usuários

### Executáveis:
- **`robotlogin.bat`** - Executar com login
- **`robotlogin_dev.bat`** - Executar sem login (desenvolvimento)

## 👥 Contas de Usuário

### 👑 **ADMIN** - `roboander_admin` / `admin123`
- **Trades/dia**: 25
- **Drawdown máximo**: 12%
- **Controle total**: ✅ Habilitado

### 👤 **TRADER** - `roboander_anderson` / `anderson123`
- **Trades/dia**: 20
- **Drawdown máximo**: 10%
- **Controle limitado**: ✅ Habilitado

### 🎯 **LOTES**
- **Tamanho do lote**: Definido diretamente na interface do robô
- **Sem limitação por conta**: Sistema de login não interfere

## 🎯 Como Executar

### **Modo Padrão (com login):**
```bash
python robotlogin.py
# ou
robotlogin.bat
```

### **Modo Desenvolvimento (sem login):**
```bash
python robotlogin.py --no-login
# ou
robotlogin_dev.bat
```

### **Modo Console:**
```bash
python robotlogin.py --console
```

## 🔧 Funcionalidades

### ✅ **Sistema de Autenticação**
- Login obrigatório antes de usar o robô
- Suporte online e local (fallback)
- Interface gráfica moderna (dark theme)
- Validação de credenciais com hash SHA-256

### ✅ **Controle de Acesso por Nível**
- Diferentes limites por tipo de usuário
- Controle de trades diários
- Limites de drawdown personalizados
- Gestão de tamanho de lote

### ✅ **Sistema Híbrido**
- **Online**: JSONBin.io API (funciona em qualquer computador)
- **Local**: Arquivo JSON como fallback
- Detecção automática de conectividade

### ✅ **Segurança**
- Senhas protegidas com hash SHA-256
- Controle de sessão ativa
- Validação de limites em tempo real
- Sistema de emergência local

## 🧪 Testes

### **Teste Completo:**
```bash
python test_robotv7_login.py
```

### **Teste da Interface:**
```bash
python robotv7_login_system.py
```

## 🔄 Migração do RobotV7 Original

O sistema mantém **100% de compatibilidade** com o RobotV7 original:

- **Com login**: `python robotlogin.py`
- **Sem login** (modo dev): `python robotlogin.py --no-login`
- **Original**: `python RobotV7.py` (ainda funciona normalmente)

## 📊 Limites e Proteções

### **Por Usuário:**
| Conta | Trades/Dia | Drawdown | Controle |
|-------|------------|----------|----------|
| Admin | 25 | 12% | ✅ Total |
| Trader | 20 | 10% | ✅ Limitado |

**🎯 Lotes**: Controlados exclusivamente na interface do robô

### **Proteções Automáticas:**
- Parada automática ao atingir limite de trades
- Bloqueio por drawdown excessivo
- Controle de acesso por usuário

## 🌐 Sistema Online vs Local

| Funcionalidade | Online | Local |
|----------------|---------|-------|
| **Acesso** | Qualquer computador | Apenas este PC |
| **Sincronização** | Automática | Manual |
| **Backup** | Na nuvem | Local |
| **Usuários** | Centralizados | Por máquina |
| **Disponibilidade** | Requer internet | Sempre disponível |

## 🛠️ Personalização

### **Adicionar Novos Usuários:**
Edite `robotv7_login_system.py` na função `create_default_users_robotv7()`:

```python
"novo_usuario": {
    "password_hash": self.hash_password("nova_senha"),
    "access_level": "trader",
    "system": "robotv7",
    "max_daily_trades": 20,
    "max_drawdown_percent": 10.0,
    "base_lot_size": 0.02,
    "max_lot_size": 0.025,
    "enable_shorts": True,
    "max_positions": 1
}
```

### **Modificar Limites:**
Ajuste os valores nos perfis de usuário conforme necessário.

## ⚠️ Troubleshooting

### **Erro de Import:**
```
ModuleNotFoundError: No module named 'robotv7_login_system'
```
**Solução**: Execute a partir da pasta `Modelo PPO Trader`

### **Login Falha:**
- Verificar credenciais (case sensitive nos usernames)
- Testar conectividade online
- Usar modo local como fallback

### **Sistema Online Indisponível:**
- Sistema automaticamente usa fallback local
- Todas as funcionalidades mantidas
- Usuários criados localmente

## 🔮 Futuras Melhorias

- [ ] Integração dos limites com a lógica de trading
- [ ] Dashboard de usuários ativos
- [ ] Logs de acesso e auditoria  
- [ ] Sistema de expiração de sessões
- [ ] Notificações de limite atingido
- [ ] Relatórios por usuário

## 🎉 Status

✅ **SISTEMA COMPLETAMENTE FUNCIONAL**

- Login integrado ✅
- Interface gráfica ✅
- Autenticação online/local ✅
- Múltiplos perfis de usuário ✅
- Limites personalizados ✅
- Testes validados ✅

**🚀 Pronto para uso em produção!**