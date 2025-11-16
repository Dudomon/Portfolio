# Trading Framework - Sistema Modular de Trading com RL

## 📁 Estrutura do Framework

```
trading_framework/
├── __init__.py                 # Imports principais do framework
├── README.md                   # Esta documentação
│
├── policies/                   # 🎯 Políticas de RL
│   ├── __init__.py
│   └── two_head_policy.py     # Política com duas cabeças (estratégica/tática)
│
├── extractors/                 # 🔍 Feature Extractors
│   ├── __init__.py
│   └── transformer_extractor.py # Extrator baseado em Transformer
│
├── rewards/                    # 🎁 Sistemas de Recompensa
│   ├── __init__.py
│   └── reward_system.py       # Sistema modular de recompensas
│
├── environments/               # 🌍 Ambientes de Trading
│   └── __init__.py
│
├── models/                     # 🤖 Modelos de ML/RL
│   └── __init__.py
│
├── utils/                      # 🛠️ Utilitários
│   └── __init__.py
│
└── configs/                    # ⚙️ Configurações
    ├── __init__.py
    └── default_configs.py     # Configurações padrão
```

## 🚀 Como Usar

### Importação Básica

```python
# Importar componentes principais
from trading_framework import TwoHeadPolicy, TransformerFeatureExtractor
from trading_framework.rewards import create_reward_system, GENTLE_GUIDANCE_CONFIG
from trading_framework.configs import get_config

# Usar configurações padrão
policy_config = get_config('policy')
ppo_config = get_config('ppo')
```

### Exemplo de Uso Completo

```python
from trading_framework.policies import TwoHeadPolicy
from trading_framework.extractors import TransformerFeatureExtractor
from trading_framework.rewards import create_reward_system, GENTLE_GUIDANCE_CONFIG
from trading_framework.configs import get_config

# Configurações
policy_config = get_config('policy')
ppo_config = get_config('ppo')

# Criar modelo PPO com componentes do framework
model = RecurrentPPO(
    policy=TwoHeadPolicy,
    env=env,
    **ppo_config,
    policy_kwargs=policy_config
)

# Sistema de recompensas
reward_system = create_reward_system("gentle_guidance", 1000, GENTLE_GUIDANCE_CONFIG)
```

## 📦 Componentes Principais

### 1. TwoHeadPolicy
Política customizada com duas cabeças:
- **Estratégica**: Decisões de alto nível (LONG/SHORT/HOLD)
- **Tática**: Gestão de posições (SL/TP, ajustes)

**Features:**
- Suporte a dropout configurável
- Compatibilidade com AMP (Automatic Mixed Precision)
- Arquitetura otimizada para trading

### 2. TransformerFeatureExtractor
Feature extractor baseado em arquitetura Transformer:
- Processamento de sequências temporais
- Attention mechanism para padrões de mercado
- Dimensões configuráveis

### 3. Sistema de Recompensas Modular
Três sistemas disponíveis:
- **Classic**: Sistema básico de recompensas
- **Balanced**: Sistema balanceado
- **Gentle Guidance**: Anti-overtrading com orientação suave

### 4. Configurações Centralizadas
Sistema de configuração unificado:
- Configurações padrão para todos os componentes
- Fácil customização e override
- Validação de parâmetros

## 🔧 Configuração

### Configurações Padrão

```python
from trading_framework.configs import get_config, update_config

# Ver todas as configurações
all_configs = get_config()

# Configuração específica
policy_config = get_config('policy')
ppo_config = get_config('ppo')

# Atualizar configuração
update_config('policy', {'policy_dropout': 0.3})
```

### Configurações Disponíveis

- `policy`: Configurações da TwoHeadPolicy
- `environment`: Configurações do ambiente de trading
- `ppo`: Hiperparâmetros do PPO
- `transformer`: Configurações do TransformerExtractor
- `optimization`: Configurações de otimização
- `evaluation`: Configurações de avaliação
- `logging`: Configurações de logging
- `checkpoint`: Configurações de checkpointing
- `metrics`: Configurações de métricas

## 🎯 Vantagens da Modularização

### ✅ Organização
- Código bem estruturado e fácil de navegar
- Separação clara de responsabilidades
- Facilita manutenção e debugging

### ✅ Reutilização
- Componentes podem ser usados independentemente
- Fácil integração em novos projetos
- Reduz duplicação de código

### ✅ Extensibilidade
- Fácil adicionar novos componentes
- Sistema de plugins natural
- Configurações centralizadas

### ✅ Testabilidade
- Cada módulo pode ser testado isoladamente
- Mocks e stubs mais simples
- Testes unitários mais focados

## 🔄 Migração de Código Existente

### Antes (código monolítico):
```python
from mainppo1 import TwoHeadPolicy
from transformer_extractor import TransformerFeatureExtractor
from reward_system import create_reward_system
```

### Depois (framework modular):
```python
from trading_framework.policies import TwoHeadPolicy
from trading_framework.extractors import TransformerFeatureExtractor
from trading_framework.rewards import create_reward_system
```

## 📈 Próximos Passos

1. **Ambientes Modulares**: Mover TradingEnv para `environments/`
2. **Utilitários**: Organizar funções auxiliares em `utils/`
3. **Modelos**: Adicionar modelos pré-treinados em `models/`
4. **Testes**: Criar suite de testes para cada módulo
5. **Documentação**: Expandir documentação com exemplos

## 🤝 Contribuição

Para adicionar novos componentes:

1. Criar módulo na pasta apropriada
2. Adicionar imports no `__init__.py` correspondente
3. Atualizar configurações se necessário
4. Documentar o novo componente
5. Adicionar testes

## 📝 Notas de Versão

### v1.0.0
- ✅ Modularização completa da TwoHeadPolicy
- ✅ Modularização do TransformerFeatureExtractor
- ✅ Modularização do sistema de recompensas
- ✅ Sistema de configurações centralizadas
- ✅ Estrutura de pastas organizada
- ✅ Documentação inicial 