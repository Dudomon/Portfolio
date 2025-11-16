#!/usr/bin/env python3
"""
🎓 CURRICULUM LEARNING V1 - 1 MINUTE PHASE
Primeira fase do curriculum learning: Micro-scalping com dados de 1 minuto

OBJETIVO DESTA FASE:
- Ensinar reações básicas de mercado (RSI, BB, momentum)
- Treinar disciplina de SL/TP pequenos (3-8 pontos)  
- Desenvolver reflexos rápidos para reversões
- Base sólida antes de expandir para timeframes maiores
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

# Importar o ambiente base do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trading_framework.envs.trading_env_v2 import TradingEnvironmentV2

def setup_logging():
    """Setup logging for curriculum learning"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [CURRICULUM-1M] - %(message)s'
    )
    return logging.getLogger(__name__)

class CurriculumPhase1Environment(TradingEnvironmentV2):
    """
    🎓 Ambiente especializado para Fase 1 do Curriculum Learning
    Foco: Micro-scalping com timeframe de 1 minuto
    """
    
    def __init__(self, **kwargs):
        # Configurações específicas para Fase 1
        curriculum_config = {
            'max_episode_steps': 500,        # Episódios mais curtos para 1m
            'lookback_window_size': 20,      # Janela menor para 1m
            'initial_balance': 500.0,        # Balance inicial menor
            'max_positions': 2,              # Máximo 2 posições para começar
            'reward_scaling': 1.5,           # Reward maior para incentivar ação
        }
        
        # Merge com configurações passadas
        kwargs.update(curriculum_config)
        super().__init__(**kwargs)
        
        # 🎯 CONFIGURAÇÕES ESPECÍFICAS PARA 1M SCALPING
        self.setup_curriculum_phase1()
        
    def setup_curriculum_phase1(self):
        """Configurações específicas da Fase 1"""
        self.logger = logging.getLogger(__name__)
        
        # SL/TP ranges para micro-scalping
        self.curriculum_sltp_config = {
            'sl_min_points': 3,      # SL mínimo: 3 pontos ($3)
            'sl_max_points': 8,      # SL máximo: 8 pontos ($8)
            'tp_min_points': 4,      # TP mínimo: 4 pontos ($4)
            'tp_max_points': 12,     # TP máximo: 12 pontos ($12)
        }
        
        # Thresholds mais permissivos para Phase 1
        self.curriculum_thresholds = {
            'confidence_min': 0.15,   # Muito permissivo para aprender
            'quality_min': 0.20,      # Aceitar setups simples
            'risk_max': 0.8,          # Aceitar mais risco para aprender
        }
        
        # Reward bonuses específicos para 1m
        self.scalping_bonuses = {
            'quick_profit_bonus': 0.5,     # Bônus para lucros rápidos (< 5min)
            'tight_sl_bonus': 0.3,         # Bônus para SL apertado usado
            'reversal_catch_bonus': 0.8,   # Bônus para pegar reversões
            'overtrading_penalty': -0.2,   # Penalidade por overtrading
        }
        
        self.logger.info("🎓 Curriculum Phase 1 (1M) configurado:")
        self.logger.info(f"   SL/TP Range: {self.curriculum_sltp_config}")
        self.logger.info(f"   Thresholds: {self.curriculum_thresholds}")
        
    def calculate_curriculum_reward(self, action, trade_result=None):
        """
        🎯 Sistema de reward específico para Curriculum Phase 1
        Foco em ensinar conceitos básicos de scalping
        """
        base_reward = 0.0
        
        if trade_result is None:
            return base_reward
            
        # 🚀 BÔNUS PARA TRADES RÁPIDOS E EFICIENTES
        trade_duration_minutes = trade_result.get('duration_minutes', 0)
        profit_points = trade_result.get('profit_points', 0)
        
        # Bonus para scalping rápido (1-10 minutos)
        if 1 <= trade_duration_minutes <= 10 and profit_points > 0:
            quick_bonus = self.scalping_bonuses['quick_profit_bonus']
            base_reward += quick_bonus
            self.logger.debug(f"🏃 Quick scalp bonus: +{quick_bonus}")
        
        # Bonus para SL efetivo (< 6 pontos)
        sl_points = trade_result.get('sl_points', 0)
        if 0 < sl_points <= 6:
            sl_bonus = self.scalping_bonuses['tight_sl_bonus']
            base_reward += sl_bonus
            self.logger.debug(f"🛡️ Tight SL bonus: +{sl_bonus}")
        
        # Bonus massivo para reversões bem executadas
        if self.detect_reversal_catch(trade_result):
            reversal_bonus = self.scalping_bonuses['reversal_catch_bonus']
            base_reward += reversal_bonus
            self.logger.debug(f"🔄 Reversal catch bonus: +{reversal_bonus}")
        
        # Penalidade por overtrading (> 5 trades em 1 hora)
        recent_trades = self.get_recent_trades_count(60)  # última hora
        if recent_trades > 5:
            overtrading_penalty = self.scalping_bonuses['overtrading_penalty']
            base_reward += overtrading_penalty
            self.logger.debug(f"⚠️ Overtrading penalty: {overtrading_penalty}")
        
        return base_reward
    
    def detect_reversal_catch(self, trade_result):
        """Detecta se o trade pegou uma reversão bem executada"""
        # Lógica simples: RSI extremo + profit em direção contrária
        entry_rsi = trade_result.get('entry_rsi', 50)
        trade_direction = trade_result.get('direction', 0)
        profit_points = trade_result.get('profit_points', 0)
        
        # Reversão de oversold (RSI < 25, long trade, profit > 4 pontos)
        if entry_rsi < 25 and trade_direction > 0 and profit_points > 4:
            return True
            
        # Reversão de overbought (RSI > 75, short trade, profit > 4 pontos) 
        if entry_rsi > 75 and trade_direction < 0 and profit_points > 4:
            return True
            
        return False
    
    def get_recent_trades_count(self, minutes_back):
        """Conta trades recentes nos últimos X minutos"""
        # Implementação simplificada
        return len([t for t in getattr(self, 'recent_trades', []) 
                   if t.get('minutes_ago', 0) <= minutes_back])

def load_1m_dataset():
    """Carrega o dataset de 1 minuto criado anteriormente"""
    logger = logging.getLogger(__name__)
    
    # Procurar pelo arquivo mais recente
    data_dir = "data"
    pkl_files = [f for f in os.listdir(data_dir) if f.startswith("GOLD_1M_CURRICULUM") and f.endswith(".pkl")]
    
    if not pkl_files:
        raise FileNotFoundError("❌ Nenhum dataset 1M encontrado! Execute download_yahoo_gold_1m.py primeiro.")
    
    # Pegar o mais recente
    latest_file = sorted(pkl_files)[-1]
    pkl_path = os.path.join(data_dir, latest_file)
    
    logger.info(f"📊 Carregando dataset 1M: {latest_file}")
    df = pd.read_pickle(pkl_path)
    
    # Validações básicas
    if len(df) < 100:
        raise ValueError(f"❌ Dataset muito pequeno: {len(df)} barras")
    
    logger.info(f"✅ Dataset carregado: {len(df)} barras, {len(df.columns)} colunas")
    logger.info(f"📅 Período: {df['time'].min()} até {df['time'].max()}")
    
    return df

def create_curriculum_env():
    """Cria ambiente otimizado para Curriculum Phase 1"""
    logger = logging.getLogger(__name__)
    
    # Carregar dados 1m
    df = load_1m_dataset()
    
    # Configurações específicas para Phase 1
    env_config = {
        'df': df,
        'initial_balance': 500.0,
        'max_episode_steps': 400,        # Episódios de ~6-7 horas de trading
        'lookback_window_size': 20,      # 20 minutos de histórico
        'reward_scaling': 2.0,           # Rewards amplificados para aprendizado
    }
    
    # Criar ambiente
    env = CurriculumPhase1Environment(**env_config)
    
    # Wrap em DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    logger.info("🎓 Ambiente Curriculum Phase 1 criado!")
    logger.info(f"   Action space: {env.action_space}")
    logger.info(f"   Observation space: {env.observation_space.shape}")
    
    return vec_env

def train_curriculum_phase1(total_timesteps=50000):
    """
    🎓 Treinar Fase 1 do Curriculum Learning
    Foco: Micro-scalping de 1 minuto
    """
    logger = setup_logging()
    
    logger.info("🚀 INICIANDO CURRICULUM LEARNING - FASE 1 (1M SCALPING)")
    logger.info("=" * 70)
    
    # Criar ambiente
    env = create_curriculum_env()
    
    # Configurações do modelo para Phase 1
    model_config = {
        'policy': 'MlpLstmPolicy',
        'env': env,
        'learning_rate': 3e-4,
        'n_steps': 256,              # Steps menores para 1m
        'batch_size': 32,            # Batch menor
        'n_epochs': 4,               # Menos epochs por update
        'gamma': 0.95,               # Gamma menor = foco em rewards imediatos
        'gae_lambda': 0.9,           # GAE menor para scalping
        'clip_range': 0.15,          # Clip range menor = atualizações mais conservadoras
        'verbose': 1,
        'device': 'cpu'
    }
    
    # Criar modelo
    logger.info("🧠 Criando modelo PPO para Curriculum Phase 1...")
    model = RecurrentPPO(**model_config)
    
    # Callback para monitoramento
    def curriculum_callback(locals_, globals_):
        if locals_['self'].num_timesteps % 5000 == 0:
            step = locals_['self'].num_timesteps
            logger.info(f"🎯 Curriculum Phase 1 - Step {step}/{total_timesteps}")
        return True
    
    # Treinar
    logger.info(f"🏋️ Iniciando treinamento - {total_timesteps} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=curriculum_callback,
        reset_num_timesteps=True
    )
    
    # Salvar modelo
    model_dir = "models/curriculum"
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"curriculum_phase1_1m_{timestamp}.zip")
    
    model.save(model_path)
    logger.info(f"💾 Modelo Curriculum Phase 1 salvo: {model_path}")
    
    # Estatísticas finais
    logger.info("🏆 CURRICULUM PHASE 1 CONCLUÍDA!")
    logger.info(f"   Modelo salvo: {model_path}")
    logger.info(f"   Total timesteps: {total_timesteps}")
    logger.info(f"   Próxima fase: Expandir para 5m + 1m")
    
    return model, model_path

def main():
    """Função principal"""
    try:
        # Treinar Fase 1 do Curriculum
        model, model_path = train_curriculum_phase1(total_timesteps=50000)
        
        print("✅ Curriculum Learning Phase 1 concluída com sucesso!")
        print(f"📁 Modelo salvo em: {model_path}")
        print("🎯 Próximo passo: Criar Phase 2 (1m + 5m)")
        
    except Exception as e:
        print(f"❌ Erro no Curriculum Learning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()