"""
Módulo de Recompensas para Trading PPO - VERSÃO COMPLETA

Exporta o sistema de rewards com lógica de execução de ordens integrada.
"""

# Importações do sistema de rewards - MÚLTIPLAS VERSÕES
try:
    from .reward_system_simple import (
        create_simple_reward_system,
        SimpleRewardCalculator, 
        SIMPLE_REWARD_CONFIG
    )
    from .clean_reward import CleanRewardCalculator, CLEAN_REWARD_CONFIG
    from .reward_daytrade_v4_inno import InnovativeMoneyReward
    from .reward_daytrade_v4_selective import SelectiveTradingReward
    from .reward_daytrade_v5_sharpe import SharpeOptimizedRewardV5
    from .reward_daytrade_v6_pro import create_v6_pro_reward_system
    print("OK Sistemas de rewards importados com sucesso")
except ImportError as e:
    print(f"AVISO Erro ao importar sistemas de rewards: {e}")
    create_simple_reward_system = None
    SimpleRewardCalculator = None
    CleanRewardCalculator = None
    InnovativeMoneyReward = None
    SelectiveTradingReward = None
    SharpeOptimizedRewardV5 = None
    create_v6_pro_reward_system = None

# Configuração para ADAPTIVE_ANTI_OVERTRADING
ADAPTIVE_ANTI_OVERTRADING_CONFIG = {
    "enable_regime_detection": False,
    "enable_portfolio_scaling": False,
    "enable_timing_evaluation": True,
    "enable_volatility_adjustment": False,
    "max_trades_per_day": 18,
    "min_trade_duration": 1,
    "quality_over_quantity": True,
    "hold_tolerance": 100,
    "scalping_penalty": 0.0,
    "target_trades_per_day": 0,  # 🚨 DESABILITADO: Sem target de trades
    "momentum_threshold": 0.000679,
    "volatility_min": 0.000577,
    "volatility_max": 0.0144
}

# Função principal para criar reward systems
def create_reward_system(reward_type="simple", initial_balance=1000.0, config=None):
    """
    Cria sistema de rewards baseado no tipo especificado
    
    Tipos disponíveis:
    - simple: Sistema básico
    - clean: Sistema limpo
    - v4_inno: V4 INNO Money-focused (PnL otimizado)
    - v4_selective: V4 SELECTIVE Anti-overtrading (Seletividade e Qualidade)
    - v5_sharpe: V5 Sharpe-Optimized (Sharpe ratio direto)
    - v6 / v6_pro: V6 Pro (PnL dominante + risco profissional)
    """
    if reward_type == "simple" and create_simple_reward_system:
        return create_simple_reward_system(initial_balance)
    elif reward_type == "clean" and CleanRewardCalculator:
        return CleanRewardCalculator(initial_balance, config or CLEAN_REWARD_CONFIG)
    elif reward_type == "v4_inno" and InnovativeMoneyReward:
        return InnovativeMoneyReward(initial_balance)
    elif reward_type == "v4_selective" and SelectiveTradingReward:
        return SelectiveTradingReward(initial_balance)
    elif reward_type == "v5_sharpe" and SharpeOptimizedRewardV5:
        return SharpeOptimizedRewardV5(initial_balance)
    elif reward_type in ("v6", "v6_pro") and create_v6_pro_reward_system:
        return create_v6_pro_reward_system(initial_balance, config)
    else:
        print(f"AVISO Tipo de reward system nao disponivel: {reward_type}")
        return None

# Exportações
__all__ = [
    'create_reward_system',
    'create_simple_reward_system',
    'SimpleRewardCalculator',
    'CleanRewardCalculator',
    'InnovativeMoneyReward',
    'SelectiveTradingReward',
    'SharpeOptimizedRewardV5',
    'create_v6_pro_reward_system',
    'SIMPLE_REWARD_CONFIG',
    'CLEAN_REWARD_CONFIG',
    'ADAPTIVE_ANTI_OVERTRADING_CONFIG',
] 
