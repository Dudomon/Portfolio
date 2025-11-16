"""
Configurações Padrão do Trading Framework
========================================

Este módulo contém todas as configurações padrão para os diferentes
componentes do sistema de trading.
"""

# Configurações de Política TwoHeadPolicy
DEFAULT_POLICY_CONFIG = {
    'features_extractor_kwargs': {'features_dim': 64},
    'lstm_hidden_size': 64,
    'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}],
    'n_lstm_layers': 1,
    'shared_lstm': False,
    'enable_critic_lstm': True,
    'policy_dropout': 0.2,
    'value_dropout': 0.1,
    'lstm_kwargs': {'batch_first': True, 'dropout': 0.2}
}

# Configurações de Ambiente de Trading
DEFAULT_TRADING_ENV_CONFIG = {
    'window_size': 20,
    'initial_balance': 1000,
    'max_lot_size': 0.08,
    'max_positions': 3,
    'max_steps': 50000
}

# Configurações de Treinamento PPO
DEFAULT_PPO_CONFIG = {
    'learning_rate': 2.22e-05,
    'n_steps': 1792,
    'batch_size': 80,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.112,
    'ent_coef': 0.0125,
    'vf_coef': 0.652,
    'max_grad_norm': 0.5
}

# Configurações de Feature Extractor
DEFAULT_TRANSFORMER_CONFIG = {
    'features_dim': 64,
    'n_heads': 8,
    'n_layers': 2,
    'dropout': 0.1,
    'activation': 'relu'
}

# Configurações de Otimização
DEFAULT_OPTIMIZATION_CONFIG = {
    'n_trials': 100,
    'refinement_trials': 25,
    'timeout_per_trial': 3600,  # 1 hora
    'pruner': 'median',
    'sampler': 'tpe'
}

# Configurações de Avaliação
DEFAULT_EVALUATION_CONFIG = {
    'eval_episodes': 10,
    'eval_freq': 10000,
    'deterministic': True,
    'render': False
}

# Configurações de Logging
DEFAULT_LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'trading_framework.log',
    'console_output': True
}

# Configurações de Checkpointing
DEFAULT_CHECKPOINT_CONFIG = {
    'save_freq': 10000,
    'keep_best_n': 3,
    'save_path': 'checkpoints',
    'auto_save': True
}

# Configurações de Métricas
DEFAULT_METRICS_CONFIG = {
    'calculate_sharpe': True,
    'calculate_sortino': True,
    'calculate_calmar': True,
    'calculate_max_drawdown': True,
    'calculate_win_rate': True,
    'calculate_profit_factor': True,
    'rolling_window': 100
}

# Configuração Completa do Framework
FRAMEWORK_CONFIG = {
    'policy': DEFAULT_POLICY_CONFIG,
    'environment': DEFAULT_TRADING_ENV_CONFIG,
    'ppo': DEFAULT_PPO_CONFIG,
    'transformer': DEFAULT_TRANSFORMER_CONFIG,
    'optimization': DEFAULT_OPTIMIZATION_CONFIG,
    'evaluation': DEFAULT_EVALUATION_CONFIG,
    'logging': DEFAULT_LOGGING_CONFIG,
    'checkpoint': DEFAULT_CHECKPOINT_CONFIG,
    'metrics': DEFAULT_METRICS_CONFIG
}

def get_config(component: str = None):
    """
    Retorna configuração para um componente específico ou toda a configuração
    
    Args:
        component: Nome do componente ('policy', 'environment', etc.)
                  Se None, retorna toda a configuração
    
    Returns:
        dict: Configuração solicitada
    """
    if component is None:
        return FRAMEWORK_CONFIG.copy()
    
    if component not in FRAMEWORK_CONFIG:
        raise ValueError(f"Componente '{component}' não encontrado. "
                        f"Disponíveis: {list(FRAMEWORK_CONFIG.keys())}")
    
    return FRAMEWORK_CONFIG[component].copy()

def update_config(component: str, updates: dict):
    """
    Atualiza configuração de um componente
    
    Args:
        component: Nome do componente
        updates: Dicionário com atualizações
    """
    if component not in FRAMEWORK_CONFIG:
        raise ValueError(f"Componente '{component}' não encontrado")
    
    FRAMEWORK_CONFIG[component].update(updates)

def reset_config(component: str = None):
    """
    Reseta configuração para valores padrão
    
    Args:
        component: Nome do componente ou None para resetar tudo
    """
    if component is None:
        # Reset completo - reimportar valores padrão
        global FRAMEWORK_CONFIG
        FRAMEWORK_CONFIG = {
            'policy': DEFAULT_POLICY_CONFIG.copy(),
            'environment': DEFAULT_TRADING_ENV_CONFIG.copy(),
            'ppo': DEFAULT_PPO_CONFIG.copy(),
            'transformer': DEFAULT_TRANSFORMER_CONFIG.copy(),
            'optimization': DEFAULT_OPTIMIZATION_CONFIG.copy(),
            'evaluation': DEFAULT_EVALUATION_CONFIG.copy(),
            'logging': DEFAULT_LOGGING_CONFIG.copy(),
            'checkpoint': DEFAULT_CHECKPOINT_CONFIG.copy(),
            'metrics': DEFAULT_METRICS_CONFIG.copy()
        }
    else:
        if component not in FRAMEWORK_CONFIG:
            raise ValueError(f"Componente '{component}' não encontrado")
        
        # Reset específico
        defaults = {
            'policy': DEFAULT_POLICY_CONFIG,
            'environment': DEFAULT_TRADING_ENV_CONFIG,
            'ppo': DEFAULT_PPO_CONFIG,
            'transformer': DEFAULT_TRANSFORMER_CONFIG,
            'optimization': DEFAULT_OPTIMIZATION_CONFIG,
            'evaluation': DEFAULT_EVALUATION_CONFIG,
            'logging': DEFAULT_LOGGING_CONFIG,
            'checkpoint': DEFAULT_CHECKPOINT_CONFIG,
            'metrics': DEFAULT_METRICS_CONFIG
        }
        
        FRAMEWORK_CONFIG[component] = defaults[component].copy()

def get_default_configs():
    """
    Retorna todas as configurações padrão do sistema
    
    Returns:
        dict: Todas as configurações padrão
    """
    return FRAMEWORK_CONFIG.copy() 