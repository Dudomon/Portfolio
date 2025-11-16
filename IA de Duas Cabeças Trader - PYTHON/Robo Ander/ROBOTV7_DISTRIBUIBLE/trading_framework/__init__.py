#!/usr/bin/env python3
"""
 TRADING FRAMEWORK - FRAMEWORK MODULARIZADO PARA TRADING ALGORÍTMICO

Framework completo para desenvolvimento de sistemas de trading usando Reinforcement Learning.
Inclui ambientes, políticas, extractors de features e utilitários otimizados.
"""

#  IMPORTS PRINCIPAIS
import os
import sys

try:
    from .environments.trading_env import TradingEnv
except ImportError as e:
    print(f"AVISO Erro ao importar TradingEnv: {e}")
    TradingEnv = None

try:
    from .policies.two_head_policy import TwoHeadPolicy
    from .policies.two_head_v2 import TwoHeadV2Policy
except ImportError as e:
    print(f"AVISO Erro ao importar políticas: {e}")
    TwoHeadPolicy = None
    TwoHeadV2Policy = None

try:
    from .extractors.transformer_extractor import TradingTransformerFeatureExtractor
except ImportError as e:
    print(f"AVISO Erro ao importar extractors: {e}")
    TradingTransformerFeatureExtractor = None

#  IMPORTS DE SISTEMAS DE REWARD
try:
    from .rewards.reward_system_simple import create_simple_reward_system, SIMPLE_REWARD_CONFIG
    from .rewards.clean_reward import CleanRewardCalculator, CLEAN_REWARD_CONFIG
    print("OK Sistemas de rewards importados com sucesso")
except ImportError as e:
    print(f"AVISO Erro ao importar sistemas de rewards: {e}")
    create_simple_reward_system = None
    CleanRewardCalculator = None

# Sistema de reward diferenciado não disponível
AdaptiveAntiOvertradingCalculator = None

#  IMPORTS DE UTILITÁRIOS
try:
    from .utils.metrics import PortfolioMetrics, TradingMetrics, MetricsFormatter
    from .utils.data_loader import DataLoader, DataValidator
    from .utils.callbacks import (
        RobustSaveCallback, 
        LearningMonitorCallback, 
        VecNormalizeCallback,
        EvaluationCallback,
        KeyboardEvaluationCallback
    )
    from .utils.config_manager import ConfigManager
    print("OK Utilitários importados")
except ImportError as e:
    print(f"AVISO Erro ao importar utilitários: {e}")
    ConfigManager = None
    PortfolioMetrics = None
    TradingMetrics = None
    MetricsFormatter = None
    DataLoader = None
    DataValidator = None
    RobustSaveCallback = None
    LearningMonitorCallback = None
    VecNormalizeCallback = None
    EvaluationCallback = None
    KeyboardEvaluationCallback = None

#  IMPORTS DE CONFIGURAÇÕES
try:
    from .configs.default_configs import get_default_configs
    print("OK Configurações padrão importadas")
except ImportError as e:
    print(f"AVISO Erro ao importar configurações: {e}")
    get_default_configs = None

#  VERSÃO DO FRAMEWORK
__version__ = "2.2.0"

#  FUNÇÃO AUXILIAR PARA CRIAR SISTEMA DE REWARD SIMPLES COM EXECUÇÃO
def create_reward_system(reward_type="simple", initial_balance=1000.0, config=None):
    """
    Cria sistema de rewards baseado no tipo especificado
    """
    if reward_type == "simple" and create_simple_reward_system:
        return create_simple_reward_system(initial_balance)
    elif reward_type == "clean" and CleanRewardCalculator:
        return CleanRewardCalculator(initial_balance, config or CLEAN_REWARD_CONFIG)
    else:
        print(f"AVISO Tipo de reward system não disponível: {reward_type}")
        return None

#  EXPORTS PRINCIPAIS
__all__ = [
    # Environments
    'TradingEnv',
    
    # Policies
    'TwoHeadPolicy',
    'TwoHeadV2Policy',
    
    # Feature Extractors
    'TradingTransformerFeatureExtractor',
    
    # Reward Systems
    'create_reward_system',
    'create_simple_reward_system',
    'CleanRewardCalculator',
    'SIMPLE_REWARD_CONFIG',
    'CLEAN_REWARD_CONFIG',
    
    # Utils
    'ConfigManager',
    'PortfolioMetrics',
    'TradingMetrics',
    'MetricsFormatter',
    'DataLoader',
    'DataValidator',
    
    # Callbacks
    'RobustSaveCallback',
    'LearningMonitorCallback',
    'VecNormalizeCallback',
    'EvaluationCallback',
    'KeyboardEvaluationCallback',
    
    # Configs
    'get_default_configs',
    
    # Version
    '__version__',
]

#  CONFIGURAÇÕES PADRÃO
DEFAULT_CONFIG = {
    # Environment
    'window_size': 20,
    'initial_balance': 1000,
    'max_positions': 3,
    
    # Policy
    'policy_type': 'TwoHeadV2Policy',  # Nova política otimizada por padrão
    
    # Features
    'features_dim': 128,
    'use_transformer': True,
    
    # Training
    'learning_rate': 2.5e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    
    # LSTM
    'lstm_hidden_size': 96,  # Otimizado para TwoHeadV2
    'n_lstm_layers': 1,      # 1 camada para atividade
    'attention_heads': 8,    # 8 heads para padrões 48h
    
    # Two Heads
    'entry_head_size': 64,
    'management_head_size': 64,
    'volatility_aware': True,
    
    # Transformer
    'seq_len': 8,            # 8 barras de histórico
    
    # Rewards
    'target_trades_per_day': 18,
    'portfolio_weight': 0.8,
    'drawdown_weight': 0.5,
    
    # VecNormalize -  DESABILITADO - USAMOS ENHANCED SCALER
    'use_vecnormalize': False,
    'norm_obs': True,
    'norm_reward': True,
    'clip_obs': 10.0,
    'clip_reward': 50.0,
}

#  FUNÇÃO PARA OBTER CONFIGURAÇÃO OTIMIZADA
def get_optimized_config():
    """Retorna configuração otimizada para TwoHeadV2Policy"""
    if not TradingTransformerFeatureExtractor:
        print("AVISO TradingTransformerFeatureExtractor não disponível")
        return DEFAULT_CONFIG
        
    return {
        **DEFAULT_CONFIG,
        'policy_kwargs': {
            'lstm_hidden_size': 96,
            'n_lstm_layers': 1,
            'attention_heads': 8,
            'entry_head_size': 64,
            'management_head_size': 64,
            'volatility_aware': True,
            'features_extractor_class': TradingTransformerFeatureExtractor,
            'features_extractor_kwargs': {
                'features_dim': 128,
                'seq_len': 8
            },
            'net_arch': [dict(pi=[128, 64, 32], vf=[128, 64, 32])],
        }
    }

#  FUNÇÃO PARA CRIAÇÃO RÁPIDA DE MODELO
def create_optimized_model(env, **kwargs):
    """Cria modelo otimizado com TwoHeadV2Policy"""
    try:
        from stable_baselines3 import PPO
        
        if not TwoHeadV2Policy:
            print("AVISO TwoHeadV2Policy não disponível")
            return None
        
        config = get_optimized_config()
        config.update(kwargs)
        
        return PPO(
            policy=TwoHeadV2Policy,
            env=env,
            **config
        )
    except ImportError as e:
        print(f"AVISO Erro ao criar modelo: {e}")
        return None

print(f" Trading Framework v{__version__} carregado com sucesso!")
if TwoHeadV2Policy:
    print(f" TwoHeadV2Policy otimizada para trading 48h disponível!")
if create_simple_reward_system:
    print(f" Sistema de reward simples disponível!")
if AdaptiveAntiOvertradingCalculator:
    print(f" Sistema de reward diferenciado disponível!")
print(f" VecNormalize configurado: obs={DEFAULT_CONFIG['norm_obs']}, reward={DEFAULT_CONFIG['norm_reward']}")

# Informações do framework
__author__ = "Trading Framework Team"
__description__ = "Framework completo para trading com PPO, métricas avançadas e sistemas de reward integrados"

#  FUNÇÃO DE INICIALIZAÇÃO RÁPIDA
def quick_start(data_file: str = None, config_preset: str = "balanced"):
    """
    Inicialização rápida do framework
    
    Args:
        data_file: Arquivo de dados CSV (opcional)
        config_preset: Preset de configuração ('conservative', 'balanced', 'aggressive')
    
    Returns:
        Tuple com (config_manager, data_loader, env)
    """
    print(" INICIALIZANDO TRADING FRAMEWORK")
    print("=" * 50)
    
    # Verificar disponibilidade dos componentes
    if not ConfigManager:
        print("AVISO ConfigManager não disponível - usando configuração padrão")
        return None, None, None
    
    # 1. Configurações
    try:
        config_manager = ConfigManager()
        if hasattr(config_manager, 'print_config_summary'):
            config_manager.print_config_summary()
    except Exception as e:
        print(f"AVISO Erro ao criar ConfigManager: {e}")
        return None, None, None
    
    # 2. Dados
    data_loader = None
    if DataLoader:
        try:
            data_loader = DataLoader()
        except Exception as e:
            print(f"AVISO Erro ao criar DataLoader: {e}")
    
    # 3. Ambiente (se dados fornecidos)
    env = None
    if data_file and TradingEnv and data_loader:
        try:
            if hasattr(data_loader, 'load_optimized_data'):
                df = data_loader.load_optimized_data(data_file)
            else:
                print("AVISO DataLoader não tem método load_optimized_data")
                return config_manager, data_loader, None
                
            env = TradingEnv(
                df=df,
                window_size=20,
                initial_balance=1000,
                is_training=True
            )
            print(f"OK Ambiente criado com {len(df)} pontos de dados")
        except Exception as e:
            print(f"AVISO Erro ao criar ambiente: {e}")
    
    print("OK Framework inicializado!")
    return config_manager, data_loader, env

#  FUNÇÃO PARA TREINAMENTO COMPLETO
def train_model(model, config_manager=None, total_timesteps: int = None):
    """
    Treina modelo com callbacks completos
    
    Args:
        model: Modelo PPO
        config_manager: Gerenciador de configurações (opcional)
        total_timesteps: Total de timesteps (opcional)
    """
    if not model:
        print("AVISO Modelo não fornecido")
        return
    
    try:
        from stable_baselines3.common.callbacks import CallbackList
        
        # Configurar timesteps
        if total_timesteps is None:
            total_timesteps = 50000  # Padrão
        
        # Callbacks disponíveis
        callbacks = []
        
        # Save callback
        if RobustSaveCallback:
            save_callback = RobustSaveCallback(
                save_freq=10000,
                save_path="checkpoints",
                name_prefix="ppo_model"
            )
            callbacks.append(save_callback)
        
        # Learning monitor
        if LearningMonitorCallback:
            learning_callback = LearningMonitorCallback(
                check_freq=1000
            )
            callbacks.append(learning_callback)
        
        # VecNormalize callback
        if VecNormalizeCallback:
            vecnorm_callback = VecNormalizeCallback(
                save_freq=10000,
                save_path="checkpoints"
            )
            callbacks.append(vecnorm_callback)
        
        # Treinar
        if callbacks:
            callback_list = CallbackList(callbacks)
            print(f" Iniciando treinamento por {total_timesteps:,} timesteps...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                progress_bar=True
            )
        else:
            print(f" Iniciando treinamento simples por {total_timesteps:,} timesteps...")
            model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True
            )
        
        print("OK Treinamento concluído!")
        
    except Exception as e:
        print(f"AVISO Erro durante treinamento: {e}")

#  FUNÇÃO PARA AVALIAÇÃO
def evaluate_model(model, env, episodes: int = 10):
    """
    Avalia modelo treinado
    
    Args:
        model: Modelo PPO treinado
        env: Ambiente de trading
        episodes: Número de episódios para avaliação
    
    Returns:
        Dict com métricas de avaliação
    """
    if not model or not env:
        print("AVISO Modelo ou ambiente não fornecido")
        return {}
    
    try:
        print(f" Avaliando modelo em {episodes} episódios...")
        
        results = []
        for episode in range(episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            results.append({
                'episode': episode + 1,
                'total_reward': episode_reward,
                'portfolio_value': getattr(env, 'portfolio_value', 0),
                'total_trades': len(getattr(env, 'trades', [])),
            })
        
        # Calcular métricas agregadas
        avg_reward = sum(r['total_reward'] for r in results) / len(results)
        avg_portfolio = sum(r['portfolio_value'] for r in results) / len(results)
        avg_trades = sum(r['total_trades'] for r in results) / len(results)
        
        evaluation_metrics = {
            'avg_reward': avg_reward,
            'avg_portfolio_value': avg_portfolio,
            'avg_trades': avg_trades,
            'episodes': episodes,
            'detailed_results': results
        }
        
        print("OK Avaliação concluída!")
        return evaluation_metrics
        
    except Exception as e:
        print(f"AVISO Erro durante avaliação: {e}")
        return {}

# Exemplo de uso
if __name__ == "__main__":
    print(" TRADING FRAMEWORK - EXEMPLO DE USO")
    print("=" * 60)
    
    # Inicialização rápida
    config_manager, data_loader, env = quick_start(config_preset="balanced")
    
    if env:
        # Criar modelo
        model = create_optimized_model(env)
        
        if model:
            # Treinar (exemplo com poucos steps)
            train_model(model, config_manager, total_timesteps=10000)
            
            # Avaliar
            results = evaluate_model(model, env, episodes=3)
            
            print("\nOK Exemplo concluído!")
        else:
            print("AVISO Não foi possível criar o modelo")
    else:
        print("AVISO Não foi possível criar o ambiente") 