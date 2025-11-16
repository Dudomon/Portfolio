"""
Configurations - Módulo de Configurações
=======================================

Este módulo contém configurações padrão e templates para
diferentes componentes do sistema de trading.

Configurações disponíveis:
- Configurações de políticas
- Configurações de ambientes
- Configurações de treinamento
- Configurações de recompensas
"""

from .default_configs import (
    get_config,
    update_config,
    reset_config,
    FRAMEWORK_CONFIG,
    DEFAULT_POLICY_CONFIG,
    DEFAULT_TRADING_ENV_CONFIG,
    DEFAULT_PPO_CONFIG,
    DEFAULT_TRANSFORMER_CONFIG
)

__all__ = [
    'get_config',
    'update_config', 
    'reset_config',
    'FRAMEWORK_CONFIG',
    'DEFAULT_POLICY_CONFIG',
    'DEFAULT_TRADING_ENV_CONFIG',
    'DEFAULT_PPO_CONFIG',
    'DEFAULT_TRANSFORMER_CONFIG'
] 