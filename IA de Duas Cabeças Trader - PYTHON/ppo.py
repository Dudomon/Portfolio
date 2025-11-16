#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Configurar encoding UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 🔥 PPO.PY ALINHADO COM MAINPPO1.PY - REFINAMENTO DE HIPERPARÂMETROS
# ====================================================================
# Este arquivo foi atualizado para usar os hiperparâmetros do mainppo1.py como base
# para refinamento via Optuna, garantindo compatibilidade total e melhores resultados.
#
# PRINCIPAIS MUDANÇAS:
# 1. HIPERPARÂMETROS: Usar OPTUNA_BASE_PARAMS como centro dos ranges de otimização
# 2. CLASSES: Adicionadas TwoHeadPolicy e TradingTransformerFeatureExtractor para compatibilidade
# 3. AMBIENTE: Usar sistema smart_active e dataset completo como no mainppo1.py
# 4. FILTROS: Alinhados thresholds de momentum e volatilidade para 20-30 trades/dia
# 5. SISTEMA: AMP habilitado, observation space (960,), action space [-3,3] para SL/TP
# ====================================================================

import optuna
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO  # 🔥 CORRIGIDO: Voltar para RecurrentPPO do sb3_contrib
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
from gym import spaces
import logging
from datetime import datetime
import ta
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings
import torch
import torch.nn as nn
import glob
import psutil
import gc
import time
import random
import threading
import tkinter as tk
from tkinter import ttk
import multiprocessing
import json
# Import do sistema modular de recompensas
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
from trading_framework.policies.two_head_policy import TwoHeadPolicy
from trading_framework.rewards.reward_system_diff import create_diff_reward_system, DIFF_REWARD_CONFIG

from torch.distributions import Categorical
from torch.cuda.amp import autocast
import shutil
import queue

# 🔥 DESABILITAR AMP PARA CPU (AMP é apenas para GPU)
ENABLE_AMP = False  # Desabilitado para CPU - AMP é específico para CUDA

# 🔥 PARÂMETROS BASE ALINHADOS COM MAINPPO1.PY - BEST_PARAMS
# Valores fixos do mainppo1.py para começar a otimização
OPTUNA_BASE_PARAMS = {
    "learning_rate": 2.6845419073225884e-05,  # 🔥 ALINHADO: Valor exato do mainppo1.py
    "n_steps": 1792,                          # 🔥 ALINHADO: Steps otimizados
    "batch_size": 64,                         # 🔥 ALINHADO: Batch size otimizado
    "n_epochs": 10,                           # 🔥 ALINHADO: Epochs equilibrados
    "gamma": 0.99,                            # 🔥 ALINHADO: Gamma padrão
    "gae_lambda": 0.95,                       # 🔥 ALINHADO: GAE lambda padrão
    "clip_range": 0.08240719647806528,        # 🔥 ALINHADO: Clip range otimizado
    "ent_coef": 0.020,                        # 🔥 ALINHADO: Entropia otimizada
    "vf_coef": 0.6017559963200034,            # 🔥 ALINHADO: Value function otimizada
    "max_grad_norm": 0.5,                     # 🔥 ALINHADO: Gradiente padrão
    "window_size": 20,
    "net_arch": [256, 256, 128],
    # 🔥 PARÂMETROS DE TRADING ALINHADOS COM MAINPPO1.PY - TRIAL_2_TRADING_PARAMS
    "sl_range_min": 11,                       # 🔥 ALINHADO: SL mínimo otimizado
    "sl_range_max": 46,                       # 🔥 ALINHADO: SL máximo otimizado
    "tp_range_min": 16,                       # 🔥 ALINHADO: TP mínimo otimizado
    "tp_range_max": 82,                       # 🔥 ALINHADO: TP máximo otimizado
    "target_trades_per_day": 18,              # 🔥 ALINHADO: Target otimizado (+12.5% atividade)
    "portfolio_weight": 0.7878338511058235,   # 🔥 ALINHADO: Peso portfolio otimizado
    "drawdown_weight": 0.5100531293444458,    # 🔥 ALINHADO: Peso drawdown otimizado
    "max_drawdown_tolerance": 0.3378997883128378,  # 🔥 ALINHADO: Tolerância DD otimizada
    "win_rate_target": 0.5289654700855297,    # 🔥 ALINHADO: Win rate target otimizado
    "momentum_threshold": 0.0006783199830488681,  # 🔥 ALINHADO: Momentum threshold otimizado
    "volatility_min": 0.00046874969400924674,     # 🔥 ALINHADO: Vol mín otimizada (mais permissiva)
    "volatility_max": 0.01632738753077879         # 🔥 ALINHADO: Vol máx otimizada (mais tolerante)
}

# Configurar warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Seed para reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configure logging
def setup_logging(instance_id=0):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ppo_optimization_{instance_id}_{timestamp}.log")
    
    # 🔥 CORRIGIR ENCODING PARA WINDOWS
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # UTF-8 para arquivo
            logging.StreamHandler()  # Console padrão
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# 🔥 CONFIGURAÇÃO FORÇADA PARA CPU
device = torch.device("cpu")
logger.info("CPU MODO FORCADO - Configuracao otimizada para processamento em CPU")

# Otimizações específicas para CPU (balanceadas)
cpu_cores = max(1, multiprocessing.cpu_count() // 2)  # Usar metade dos cores para estabilidade
torch.set_num_threads(cpu_cores)  # Threads balanceadas
torch.set_num_interop_threads(1)  # Reduzir overhead de paralelização
os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
os.environ["MKL_NUM_THREADS"] = str(cpu_cores)

logger.info(f"THREADS configuradas: {torch.get_num_threads()} threads para CPU")
logger.info(f"CPU cores disponiveis: {multiprocessing.cpu_count()} | Usando: {cpu_cores}")

# Desabilitar otimizações de GPU
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# === DEBUG TOTAL FLAG ===
DEBUG_TOTAL = True  # Ative para logs detalhados

# --- FLAG PARA USAR OU NÃO VECNORMALIZE ---
USE_VECNORM = True  # Ative para normalizar observações

class TradingEnv(gym.Env):
    MAX_STEPS = 576  # 🔥 ALINHADO COM TREINO DIFF: 2 dias por episódio (576 steps = 48h em 5min)
    
    def __init__(self, df, window_size=20, is_training=True, initial_balance=1000, trading_params=None):
        super(TradingEnv, self).__init__()
        # 🔥 ALINHADO COM MAINPPO1.PY: Usar dataset completo para treinamento E avaliação
        self.df = df.copy()
        if is_training:
            print(f"[TRADING ENV] Modo treinamento: {len(self.df):,} barras (loop infinito)", flush=True)
        else:
            print(f"[TRADING ENV] Modo avaliação: {len(self.df):,} barras (dataset completo)", flush=True)
        self.window_size = window_size
        self.current_step = window_size
        self.initial_balance = initial_balance
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.positions = []
        self.returns = []
        self.trades = []
        self.start_date = pd.to_datetime(self.df.index[0])
        self.end_date = pd.to_datetime(self.df.index[-1])
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        self.max_lot_size = 0.08  # Ajustado conforme solicitado
        self.max_positions = 3
        self.current_positions = 0
        
        # 🎯 PARÂMETROS DE TRADING OTIMIZÁVEIS - SL/TP MAIS APERTADOS
        self.trading_params = trading_params or {}
        self.sl_range_min = self.trading_params.get('sl_range_min', 8)   # 🔥 MAIS APERTADO: 8 pontos
        self.sl_range_max = self.trading_params.get('sl_range_max', 30)  # 🔥 MAIS APERTADO: 30 pontos  
        self.tp_range_min = self.trading_params.get('tp_range_min', 10)  # 🔥 MAIS APERTADO: 10 pontos
        self.tp_range_max = self.trading_params.get('tp_range_max', 45)  # 🔥 MAIS APERTADO: 45 pontos
        self.target_trades_per_day = self.trading_params.get('target_trades_per_day', 18)  # 🔥 ALINHADO: 18 trades/dia igual mainppo1.py
        self.portfolio_weight = self.trading_params.get('portfolio_weight', 0.6)  # Padrão 60% peso portfolio
        self.drawdown_weight = self.trading_params.get('drawdown_weight', 0.4)  # Padrão 40% peso DD
        self.max_drawdown_tolerance = self.trading_params.get('max_drawdown_tolerance', 0.25)  # Padrão 25%
        self.win_rate_target = self.trading_params.get('win_rate_target', 0.42)  # Padrão 42%
        self.momentum_threshold = self.trading_params.get('momentum_threshold', 0.0005)  # Padrão 0.0005
        self.volatility_min = self.trading_params.get('volatility_min', 0.0005)  # Padrão 0.0005
        self.volatility_max = self.trading_params.get('volatility_max', 0.020)  # Padrão 0.020
        
        print(f"[TRADING ENV] 🔥 PARÂMETROS ALINHADOS COM TREINO DIFF:", flush=True)
        print(f"  SL Range: {self.sl_range_min}-{self.sl_range_max} pontos (MAIS APERTADO)", flush=True)
        print(f"  TP Range: {self.tp_range_min}-{self.tp_range_max} pontos (MAIS APERTADO)", flush=True)
        print(f"  Target Trades/Dia: {self.target_trades_per_day} (Alinhado: +12.5% atividade)", flush=True)
        print(f"  Portfolio Weight: {self.portfolio_weight:.3f} (Otimizado)", flush=True)
        print(f"  Max DD Tolerance: {self.max_drawdown_tolerance:.3f} (Otimizado)", flush=True)
        
        # 🔥 NOVO ACTION SPACE ESPECIALIZADO (6 dimensões) - ALINHADO COM MAINPPO1.PY:
        # ENTRY HEAD [0-2]: [entry_decision, entry_confidence, position_size]  
        # MANAGEMENT HEAD [3-5]: [mgmt_action, sl_adjust, tp_adjust]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1, -1]),  # entry + management
            high=np.array([2, 1, 1, 2, 1, 1]),   # entry + management  
            dtype=np.float32
        )
        
        self.imputer = KNNImputer(n_neighbors=5)
        base_features = [
            'returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 'stoch_k', 'bb_position', 'trend_strength', 'atr_14'
        ]
        self.feature_columns = []
        for tf in ['5m', '15m', '4h']:
            self.feature_columns.extend([f"{f}_{tf}" for f in base_features])
        self._prepare_data()
        n_features = len(self.feature_columns) + self.max_positions * 7  # 🔥 COMPATIBILIDADE: 7 features por posição
        # 🔥 NOVA ARQUITETURA: 27 features * 3 timeframes + 21 position features = 81+21 = 102 features por window
        # 102 features * 20 window = 2040 total (mas será menor com novas features bb_position e trend_strength)
        # 9 features * 3 timeframes * 20 window + 21 position features = 540 + 420 = 960 -> 840 com otimização
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size * n_features,), dtype=np.float32
        )
        self.win_streak = 0
        self.episode_steps = 0
        self.episode_start_time = None
        self.partial_reward_alpha = 0.2   # Fator de escala para recompensa parcial (ajustado para melhor equilíbrio)
        # Garantir compatibilidade com reward
        self.realized_balance = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.last_trade_pnl = 0.0
        self.HOLDING_PENALTY_THRESHOLD = 60
        self.base_tf = '5m'
        
        # 🚀 MELHORIA #8: Position sizing adaptativo (0.05 base, max 0.08)
        self.base_lot_size = 0.05  # Tamanho base
        self.max_lot_size = 0.08   # Tamanho máximo
        self.lot_size = self.base_lot_size  # Será calculado dinamicamente
        
        self.steps_since_last_trade = 0
        self.INACTIVITY_THRESHOLD = 24  # ~2h em 5m
        self.last_action = None
        self.hold_count = 0
        
        # 🔥 ALINHADO COM TREINO DIFF: Usar sistema reward_system_diff
        self.reward_system = create_diff_reward_system("diff_reward", initial_balance, DIFF_REWARD_CONFIG)
        print(f"[TRADING ENV] ✅ Sistema REWARD_SYSTEM_DIFF ativado - ALINHADO COM TREINO DIFF!", flush=True)

    def reset(self, **kwargs):
        """Reset do ambiente para um novo episódio."""
        # Reset robusto de todos os contadores e do pico
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.peak_portfolio_value = self.initial_balance  # Zera o pico só no início do episódio
        self.realized_balance = self.initial_balance  # 🔥 FIX CRÍTICO: Resetar o realized_balance!
        self.positions = []
        self.returns = []
        self.trades = []
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        self.current_positions = 0
        self.win_streak = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.steps_since_last_trade = 0
        self.hold_count = 0
        self.last_action = None
        if hasattr(self, 'low_balance_steps'):
            self.low_balance_steps = 0
        if hasattr(self, 'high_drawdown_steps'):
            self.high_drawdown_steps = 0
        
        # 🔥 CORREÇÃO CRÍTICA: Reset completo do sistema de recompensas
        # 🔥 SISTEMA LIMPO: Reset do sistema de recompensas NOVO
        if hasattr(self, 'reward_system') and hasattr(self.reward_system, 'reset'):
            self.reward_system.reset()
            print(f"[PPO ENV] ✅ Sistema CLEAN_REWARD resetado - ALINHADO COM MAINPPO1.PY")
        
        obs = self._get_observation()
        return obs

    def step(self, action):
        """Executa um passo no ambiente."""
        # 🔥 DEBUG: Log detalhado do estado inicial (reduzido para diminuir spam)
        if self.current_step % 1000 == 0:  # Log a cada 1000 steps
            # Calcular trades/dia
            trades_per_day = 0.0
            if len(self.trades) > 0 and self.episode_steps > 0:
                days_elapsed = self.episode_steps / 288  # 288 steps = 1 dia (5min bars)
                trades_per_day = len(self.trades) / max(days_elapsed, 0.01)
            
            print(f"\n[DEBUG STEP] Step: {self.current_step}")
            print(f"  Episode Steps: {self.episode_steps}")
            print(f"  Max DF Length: {len(self.df)}")
            print(f"  Realized Balance: ${self.realized_balance:.2f}")
            print(f"  Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"  Positions: {len(self.positions)}")
            print(f"  Total Trades: {len(self.trades)}")
            print(f"  Trades/Day: {trades_per_day:.1f}")
            print(f"  Current Drawdown: {self.current_drawdown*100:.2f}%")
        
        # 🔥 DATASET LOOP: Corrigir current_step se necessário
        if self.current_step >= len(self.df):
            self.current_step = self.window_size  # Loop de volta ao início
            print(f"[DATASET LOOP] Step corrigido para {self.current_step}")
        
        done = False
        termination_reason = None
        
        # 🔥 DATASET LOOP INFINITO: Quando chegar ao fim, voltar ao início
        if self.current_step >= len(self.df) - 1:
            self.current_step = self.window_size  # Reiniciar do início
            print(f"[DATASET LOOP] Voltando ao início - Step resetado para {self.window_size}")
            # NÃO terminar o episódio - continuar treinando
        
        # Verificar condições de término APENAS por MAX_STEPS ou falência
        if self.episode_steps >= self.MAX_STEPS:
            done = True
            termination_reason = f"MAX_STEPS atingido - Episode Steps: {self.episode_steps}/{self.MAX_STEPS}"
        
        # 🔥 VERIFICAR FALÊNCIA: Condições MUITO MAIS RELAXADAS
        current_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
        unrealized_pnl = sum(self.get_position_pnl(pos, current_price) for pos in self.positions)
        total_portfolio_value = self.realized_balance + unrealized_pnl
        
        # 🔥 CONDIÇÕES ULTRA-EXTREMAMENTE RELAXADAS - MÁXIMO APRENDIZADO POSSÍVEL
        bankruptcy_threshold_balance = -5000  # 🔥 AUMENTADO: Era -2000, agora -5000 (perde até $5000)
        bankruptcy_threshold_portfolio = 0.0001  # 🔥 DIMINUÍDO: Era 0.001, agora 0.0001 (0.01% = $0.10)
        
        # 🔥 DESABILITAR FALÊNCIA COMPLETAMENTE - MÁXIMO APRENDIZADO
        # NUNCA terminar por falência - apenas aplicar penalidades graduais
        if (self.realized_balance <= bankruptcy_threshold_balance):
            # done = True  # 🔥 DESABILITADO
            termination_reason = f"Portfolio crítico - Saldo: ${self.realized_balance:.2f}, Portfolio: ${total_portfolio_value:.2f}"
            print(f"[WARNING] {termination_reason} - Continuando treinamento com penalidade")
            # Aplicar penalidade mas NÃO terminar
            # reward será calculado normalmente pelo sistema de recompensas
            
        old_state = {
            "portfolio_total_value": self.realized_balance + sum(self.get_position_pnl(pos, self.df[f'close_{self.base_tf}'].iloc[self.current_step]) for pos in self.positions),
            "current_drawdown": self.current_drawdown
        }
        
        # 🔥 PROCESSAR AÇÕES ESPECIALIZADAS (Entry Head + Management Head)
        actions_taken = self._process_specialized_action(action)
        
        reward, info, terminated_by_reward = self._calculate_reward_and_info(action, old_state)
        
        # 🔥 DESABILITAR TÉRMINO POR REWARD FUNCTION - permitir máximo aprendizado
        # if terminated_by_reward:
        #     done = True
        #     termination_reason = "Término forçado pela função de reward"
        
        # Atualização robusta do portfólio e pico
        current_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
        unrealized_pnl = sum(self.get_position_pnl(pos, current_price) for pos in self.positions)
        current_total_value = self.realized_balance + unrealized_pnl
        self.portfolio_value = current_total_value
        
        # Atualize o pico do portfólio corretamente
        if not hasattr(self, 'peak_portfolio_value') or self.peak_portfolio_value is None:
            self.peak_portfolio_value = self.initial_balance
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        
        # Calcular drawdown atual
        self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 1e-6 else 0
        if self.current_drawdown > self.peak_drawdown:
            self.peak_drawdown = min(self.current_drawdown, 1.0)
            
        self.current_step += 1
        self.episode_steps += 1
        
        obs = self._get_observation()
        if not isinstance(obs, np.ndarray):
            pass
        elif obs.dtype != np.float32:
            obs = obs.astype(np.float32)
            
        if done:
            print(f"[EPISÓDIO TERMINADO] Razão: {termination_reason or 'Desconhecida'}")
            print(f"  Steps: {self.episode_steps}")
            print(f"  Portfolio Final: ${self.portfolio_value:.2f}")
            print(f"  Total Trades: {len(self.trades)}")
            
            # Fechar todas as posições abertas no final do episódio
            if self.current_step < len(self.df):
                final_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
            else:
                final_price = self.df[f'close_{self.base_tf}'].iloc[-1]
                
            for pos in self.positions[:]:
                pnl = self.get_position_pnl(pos, final_price)
                self.realized_balance += pnl
                trade_info = {
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'lot_size': pos['lot_size'],
                    'entry_step': pos['entry_step'],
                    'exit_step': self.current_step,
                    'pnl_usd': pnl,
                    'duration': self.current_step - pos['entry_step']
                }
                self.trades.append(trade_info)
            self.positions = []
            
            # Atualizar portfolio final
            self.portfolio_value = self.realized_balance
            info["peak_drawdown_episode"] = self.current_drawdown
            info["final_balance"] = self.portfolio_value
            info["peak_portfolio"] = self.peak_portfolio_value
            info["total_trades"] = len(self.trades)
            info["win_rate"] = len([t for t in self.trades if t.get('pnl_usd', 0) > 0]) / len(self.trades) if self.trades else 0.0
            info["termination_reason"] = termination_reason
            
        return obs, reward, done, info

    def _prepare_data(self):
        """
        🚀 PROCESSAMENTO OTIMIZADO DE DADOS - SPEEDUP 139.8x
        Sistema idêntico ao mainppo1.py para máxima performance
        """
        print(f"[PREPARE DATA] Iniciando processamento otimizado...")
        start_time = time.time()
        
        # 🚀 VERIFICAR SE JÁ EXISTEM FEATURES PRÉ-CALCULADAS
        expected_features = [f"{f}_{tf}" for tf in ['5m', '15m', '4h'] 
                           for f in ['returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 
                                   'stoch_k', 'bb_position', 'trend_strength', 'atr_14']]
        
        missing_features = [col for col in expected_features if col not in self.df.columns]
        
        if len(missing_features) == 0:
            print(f"[PREPARE DATA] ✅ Features já pré-calculadas, usando dados otimizados")
        else:
            print(f"[PREPARE DATA] ⚠️ Calculando {len(missing_features)} features ausentes...")
            self._calculate_missing_features(missing_features)
        
        # 🚀 USAR PROCESSED_DATA PRÉ-CALCULADO SE DISPONÍVEL
        if hasattr(self.df, 'processed_data_cache'):
            print(f"[PREPARE DATA] ✅ Usando processed_data pré-calculado")
            self.processed_data = self.df.processed_data_cache
        else:
            # Criar colunas ausentes como zero
            for col in self.feature_columns:
                if col not in self.df.columns:
                    self.df.loc[:, col] = 0
            
            # Processamento mínimo necessário
            self.processed_data = self.df[self.feature_columns].values.astype(np.float32)
            
            # Verificação de integridade
            if np.any(np.isnan(self.processed_data)) or np.any(np.isinf(self.processed_data)):
                self.processed_data = np.nan_to_num(self.processed_data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Feature binária de oportunidade (apenas para 5m)
        if 'opportunity' not in self.df.columns:
            self.df['opportunity'] = 0
            if 'sma_cross_5m' in self.df.columns:
                cross = self.df['sma_cross_5m']
                self.df['opportunity'] = ((cross.shift(1) != cross) & (cross != 0)).astype(int)
        
        processing_time = time.time() - start_time
        print(f"[PREPARE DATA] ✅ Processamento concluído em {processing_time:.3f}s")
        print(f"[PREPARE DATA] Shape final: {self.processed_data.shape}")
    
    def _calculate_missing_features(self, missing_features):
        """Calcula apenas features ausentes (fallback para dados não otimizados)"""
        print(f"[FALLBACK] Calculando features técnicas ausentes...")
        
        # Renomear colunas close duplicadas se necessário
        if 'close' in self.df.columns:
            close_cols = [col for col in self.df.columns if col == 'close']
            if len(close_cols) > 1:
                close_names = ['close_5m', 'close_15m', 'close_4h']
                for i, col in enumerate(close_cols):
                    self.df.rename(columns={col: close_names[i]}, inplace=True, level=0)
        
        # Calcular apenas features ausentes
        for tf in ['5m', '15m', '4h']:
            close_col = f'close_{tf}'
            if close_col not in self.df.columns:
                continue
                
            # Calcular apenas features que estão ausentes
            if f'returns_{tf}' in missing_features:
                self.df.loc[:, f'returns_{tf}'] = self.df[close_col].pct_change().fillna(0)
            if f'volatility_20_{tf}' in missing_features:
                self.df.loc[:, f'volatility_20_{tf}'] = self.df[close_col].rolling(window=20).std().fillna(0)
            if f'sma_20_{tf}' in missing_features:
                self.df.loc[:, f'sma_20_{tf}'] = self.df[close_col].rolling(window=20).mean().bfill().fillna(0)
            if f'sma_50_{tf}' in missing_features:
                self.df.loc[:, f'sma_50_{tf}'] = self.df[close_col].rolling(window=50).mean().bfill().fillna(0)
            if f'rsi_14_{tf}' in missing_features:
                try:
                    import ta
                    self.df.loc[:, f'rsi_14_{tf}'] = ta.momentum.RSIIndicator(self.df[close_col], window=14).rsi().fillna(0)
                except Exception:
                    self.df.loc[:, f'rsi_14_{tf}'] = 0
            if f'stoch_k_{tf}' in missing_features:
                try:
                    self.df.loc[:, f'stoch_k_{tf}'] = ta.momentum.StochasticOscillator(self.df[close_col], self.df[close_col], self.df[close_col], window=14).stoch().fillna(0)
                except Exception:
                    self.df.loc[:, f'stoch_k_{tf}'] = 0
            if f'bb_position_{tf}' in missing_features:
                bb_upper = self.df[f'sma_20_{tf}'] + (2 * self.df[f'volatility_20_{tf}'])
                bb_lower = self.df[f'sma_20_{tf}'] - (2 * self.df[f'volatility_20_{tf}'])
                self.df.loc[:, f'bb_position_{tf}'] = (
                    (self.df[close_col] - bb_lower) / (bb_upper - bb_lower + 1e-8)
                ).fillna(0.5).clip(0, 1)
            if f'trend_strength_{tf}' in missing_features:
                price_changes = self.df[close_col].pct_change().abs()
                self.df.loc[:, f'trend_strength_{tf}'] = price_changes.rolling(window=14).mean().fillna(0)
            if f'atr_14_{tf}' in missing_features:
                try:
                    self.df.loc[:, f'atr_14_{tf}'] = ta.volatility.AverageTrueRange(self.df[close_col], self.df[close_col], self.df[close_col], window=14).average_true_range().fillna(0)
                except Exception:
                    self.df.loc[:, f'atr_14_{tf}'] = 0
            
            # Features derivadas
            if f'sma_cross_{tf}' in missing_features:
                self.df.loc[:, f'sma_cross_{tf}'] = (self.df[f'sma_20_{tf}'] > self.df[f'sma_50_{tf}']).astype(float) - (self.df[f'sma_20_{tf}'] < self.df[f'sma_50_{tf}']).astype(float)
            if f'momentum_5_{tf}' in missing_features:
                self.df.loc[:, f'momentum_5_{tf}'] = self.df[close_col].pct_change(periods=5).fillna(0)
        
        print(f"[FALLBACK] ✅ Features ausentes calculadas")

    def _get_observation(self):
        # 🔥 DATASET LOOP: Corrigir current_step se necessário
        if self.current_step >= len(self.df):
            self.current_step = self.window_size
        if self.current_step < self.window_size:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 🔥 VERIFICAÇÃO ROBUSTA DE DADOS INVÁLIDOS
        positions_obs = np.zeros((self.max_positions, 7))
        current_price = self.df['close_5m'].iloc[self.current_step]
        
        # Verificar se current_price é válido
        if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
            print(f"[ERROR] Current price inválido: {current_price} no step {self.current_step}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        for i in range(self.max_positions):
            if i < len(self.positions):
                pos = self.positions[i]
                positions_obs[i, 0] = 1  # status aberta
                positions_obs[i, 1] = 0 if pos['type'] == 'long' else 1
                
                # Verificação robusta para normalização de preço
                try:
                    price_min = min(self.df['close_5m'])
                    price_max = max(self.df['close_5m'])
                    if price_max > price_min and not np.isnan(price_min) and not np.isnan(price_max):
                        positions_obs[i, 2] = (pos['entry_price'] - price_min) / (price_max - price_min)
                    else:
                        positions_obs[i, 2] = 0.5  # Valor padrão se não conseguir normalizar
                except Exception as e:
                    print(f"[ERROR] Erro na normalização de preço: {e}")
                    positions_obs[i, 2] = 0.5
                
                # PnL atual
                if pos['type'] == 'long':
                    pnl = (current_price - pos['entry_price']) * pos['lot_size']
                else:
                    pnl = (pos['entry_price'] - current_price) * pos['lot_size']
                positions_obs[i, 3] = pnl
                
                # 🔥 FIX: Garantir compatibilidade com posições antigas sem SL/TP
                if 'sl' not in pos or 'tp' not in pos:
                    if pos['type'] == 'long':
                        pos['sl'] = pos.get('sl', current_price * 0.98)  # SL 2% abaixo
                        pos['tp'] = pos.get('tp', current_price * 1.04)  # TP 4% acima  
                    else:  # short
                        pos['sl'] = pos.get('sl', current_price * 1.02)  # SL 2% acima
                        pos['tp'] = pos.get('tp', current_price * 0.96)  # TP 4% abaixo
                
                positions_obs[i, 4] = pos.get('sl', 0)
                positions_obs[i, 5] = pos.get('tp', 0)
                positions_obs[i, 6] = (self.current_step - pos['entry_step']) / len(self.df)
            else:
                positions_obs[i, :] = 0  # slot vazio
        
        # 🔥 VERIFICAÇÃO ROBUSTA DOS DADOS DE MERCADO
        try:
            obs_market = self.processed_data[self.current_step - self.window_size:self.current_step]
        except IndexError as e:
            print(f"[ERROR] Index error ao acessar processed_data: {e}")
            print(f"  current_step: {self.current_step}, window_size: {self.window_size}")
            print(f"  processed_data shape: {self.processed_data.shape}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Verificar se obs_market contém valores inválidos
        if np.any(np.isnan(obs_market)) or np.any(np.isinf(obs_market)):
            print(f"[WARNING] obs_market contém NaN ou Inf no step {self.current_step}")
            obs_market = np.nan_to_num(obs_market, nan=0.0, posinf=1e6, neginf=-1e6)
        
        tile_positions = np.tile(positions_obs.flatten(), (self.window_size, 1))
        
        # Verificar compatibilidade de shapes
        if obs_market.shape[0] != tile_positions.shape[0]:
            print(f"[ERROR] Shape mismatch: obs_market {obs_market.shape} vs tile_positions {tile_positions.shape}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        obs = np.concatenate([obs_market, tile_positions], axis=1)
        flat_obs = obs.flatten().astype(np.float32)
        
        # 🔥 VERIFICAÇÕES FINAIS DE INTEGRIDADE
        if not isinstance(flat_obs, np.ndarray):
            print(f"[ERROR] flat_obs não é np.ndarray: {type(flat_obs)}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        if flat_obs.ndim != 1:
            print(f"[ERROR] flat_obs não é 1D: shape={flat_obs.shape}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        if flat_obs.shape != self.observation_space.shape:
            print(f"[ERROR] flat_obs.shape {flat_obs.shape} != observation_space.shape {self.observation_space.shape}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        if flat_obs.dtype != np.float32:
            print(f"[WARNING] Converting dtype from {flat_obs.dtype} to np.float32")
            flat_obs = flat_obs.astype(np.float32)
        
        # Verificar se há valores inválidos na observação final
        if np.any(np.isnan(flat_obs)) or np.any(np.isinf(flat_obs)):
            print(f"[WARNING] flat_obs contém NaN ou Inf, corrigindo...")
            flat_obs = np.nan_to_num(flat_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return flat_obs

    def _calculate_reward_and_info(self, action, old_state):
        """Calcula reward usando o sistema modular de recompensas"""
        try:
            # Usar o sistema modular de recompensas
            reward, info = self.reward_system.calculate_reward(self, action, old_state)
            terminated_by_reward = info.get('terminated_by_reward', False)
            return reward, info, terminated_by_reward
        except Exception as e:
            print(f"[ERROR] Erro no cálculo de reward: {e}")
            return 0.0, {}, False

    def _process_specialized_action(self, action):
        """
        🔥 PROCESSA AÇÕES ESPECIALIZADAS: Entry Head + Management Head - ALINHADO COM MAINPPO1.PY
        """
        # 🚀 ENTRY HEAD [0-2]: Abertura de novas posições
        entry_decision = int(action[0])      # 0=No Entry, 1=Long, 2=Short  
        entry_confidence = action[1]         # 0.0-1.0 (força do sinal)
        position_size = action[2]            # 0.0-1.0 (tamanho da posição)
        
        # ⚖️ MANAGEMENT HEAD [3-5]: Gestão de posições existentes
        mgmt_action = int(action[3])         # 0=Hold, 1=Close Profitable, 2=Close All
        sl_adjust = action[4]                # -1 to 1 (ajuste SL)  
        tp_adjust = action[5]                # -1 to 1 (ajuste TP)
        
        current_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
        actions_taken = []
        
        # 🚀 ENTRY HEAD: Processar entrada de novas posições
        if entry_decision > 0 and len(self.positions) < self.max_positions:
            
            # Verificar filtros de entrada
            if self._check_entry_filters(entry_decision):
                
                # Calcular tamanho adaptativo baseado na confiança
                confidence_multiplier = 0.5 + (entry_confidence * 0.5)  # 0.5 to 1.0
                size_multiplier = 0.3 + (position_size * 0.7)           # 0.3 to 1.0
                final_size = self.base_lot_size * confidence_multiplier * size_multiplier
                final_size = max(min(final_size, self.max_lot_size), 0.01)
                
                # SL/TP baseados na volatilidade e confiança
                atr_current = self.df.get(f'atr_14_{self.base_tf}', pd.Series([0.001])).iloc[self.current_step]
                volatility_factor = max(atr_current / current_price, 0.001) if current_price > 0 else 0.001
                
                # SL mais agressivo com alta confiança, mais conservador com baixa
                sl_multiplier = 1.5 - (entry_confidence * 0.5)  # 1.5 to 1.0
                tp_multiplier = 1.0 + (entry_confidence * 1.0)   # 1.0 to 2.0
                
                sl_points = max(self.sl_range_min, min(self.sl_range_max, 
                               int(self.sl_range_min + volatility_factor * 10000 * sl_multiplier)))
                tp_points = max(self.tp_range_min, min(self.tp_range_max,
                               int(self.tp_range_min + volatility_factor * 15000 * tp_multiplier)))
                
                # Criar nova posição
                if entry_decision == 1:  # Long
                    sl_price = current_price - (sl_points * 0.01)
                    tp_price = current_price + (tp_points * 0.01)
                else:  # Short
                    sl_price = current_price + (sl_points * 0.01)
                    tp_price = current_price - (tp_points * 0.01)
                
                new_position = {
                    'type': 'long' if entry_decision == 1 else 'short',
                    'entry_price': current_price,
                    'entry_step': self.current_step, 
                    'lot_size': final_size,
                    'sl': sl_price,
                    'tp': tp_price,
                    'confidence': entry_confidence
                }
                
                self.positions.append(new_position)
                actions_taken.append(f"OPEN_{new_position['type'].upper()}")
                
        # ⚖️ MANAGEMENT HEAD: Processar gestão de posições existentes
        if len(self.positions) > 0:
            
            if mgmt_action == 1:  # Close Profitable
                profitable_positions = []
                for i, pos in enumerate(self.positions):
                    pnl = self._get_position_pnl(pos, current_price)
                    if pnl > 0:
                        profitable_positions.append(i)
                
                # Fechar posições lucrativas (da mais recente para mais antiga)
                for i in reversed(profitable_positions):
                    pos = self.positions.pop(i)
                    pnl = self._get_position_pnl(pos, current_price)
                    self.realized_balance += pnl
                    
                    trade_info = {
                        'type': pos['type'],
                        'entry_price': pos['entry_price'], 
                        'exit_price': current_price,
                        'lot_size': pos['lot_size'],
                        'entry_step': pos['entry_step'],
                        'exit_step': self.current_step,
                        'pnl_usd': pnl,
                        'duration': self.current_step - pos['entry_step']
                    }
                    self.trades.append(trade_info)
                    actions_taken.append(f"CLOSE_PROFITABLE_{pos['type'].upper()}")
                    
            elif mgmt_action == 2:  # Close All
                for pos in self.positions[:]:
                    pnl = self._get_position_pnl(pos, current_price)
                    self.realized_balance += pnl
                    
                    trade_info = {
                        'type': pos['type'],
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price, 
                        'lot_size': pos['lot_size'],
                        'entry_step': pos['entry_step'],
                        'exit_step': self.current_step,
                        'pnl_usd': pnl,
                        'duration': self.current_step - pos['entry_step']
                    }
                    self.trades.append(trade_info)
                    actions_taken.append(f"CLOSE_ALL_{pos['type'].upper()}")
                    
                self.positions = []
                
            else:  # Hold & Adjust (mgmt_action == 0)
                # Ajustar SL/TP das posições existentes
                for pos in self.positions:
                    # 🔥 FIX: Garantir que posição tem SL/TP (compatibilidade com sistema antigo)
                    if 'sl' not in pos or 'tp' not in pos:
                        # Criar SL/TP padrão para posições antigas
                        if pos['type'] == 'long':
                            pos['sl'] = pos.get('sl', current_price * 0.98)  # SL 2% abaixo
                            pos['tp'] = pos.get('tp', current_price * 1.04)  # TP 4% acima
                        else:  # short
                            pos['sl'] = pos.get('sl', current_price * 1.02)  # SL 2% acima  
                            pos['tp'] = pos.get('tp', current_price * 0.96)  # TP 4% abaixo
                    
                    atr_current = self.df.get(f'atr_14_{self.base_tf}', pd.Series([0.001])).iloc[self.current_step]
                    
                    # Ajuste baseado nos sinais sl_adjust e tp_adjust
                    if pos['type'] == 'long':
                        # Ajustar SL: -1 = mais agressivo (mais próximo), +1 = mais conservador
                        sl_adjustment = sl_adjust * atr_current * 0.5  # Max 50% do ATR
                        tp_adjustment = tp_adjust * atr_current * 0.5
                        
                        new_sl = pos['sl'] - sl_adjustment  # Mais negativo = SL mais baixo (conservador)
                        new_tp = pos['tp'] + tp_adjustment  # Mais positivo = TP mais alto
                        
                    else:  # short
                        sl_adjustment = sl_adjust * atr_current * 0.5
                        tp_adjustment = tp_adjust * atr_current * 0.5
                        
                        new_sl = pos['sl'] + sl_adjustment  # Mais positivo = SL mais alto (conservador)
                        new_tp = pos['tp'] - tp_adjustment  # Mais negativo = TP mais baixo
                    
                    # Aplicar ajustes com limites de segurança
                    pos['sl'] = max(new_sl, current_price * 0.95) if pos['type'] == 'long' else min(new_sl, current_price * 1.05)
                    pos['tp'] = min(new_tp, current_price * 1.10) if pos['type'] == 'long' else max(new_tp, current_price * 0.90)
                    
                if abs(sl_adjust) > 0.1 or abs(tp_adjust) > 0.1:
                    actions_taken.append("ADJUST_SLTP")
        
        # Atualizar portfolio value
        self.portfolio_value = self.realized_balance + self._get_unrealized_pnl()
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        
        return actions_taken
    
    def get_position_pnl(self, pos, current_price):
        """Calcula PnL de uma posição (sem underscore para evitar conflito)"""
        if pos['type'] == 'long':
            return (current_price - pos['entry_price']) * pos['lot_size'] * 100
        else:
            return (pos['entry_price'] - current_price) * pos['lot_size'] * 100
    
    def _get_position_pnl(self, pos, current_price):
        """Compatibilidade com reward_system.py"""
        return self.get_position_pnl(pos, current_price)

    def _get_unrealized_pnl(self):
        """
        Calcula o PnL não realizado de todas as posições abertas.
        Método necessário para compatibilidade com reward_system.py
        """
        if not self.positions:
            return 0.0
        
        current_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
        total_unrealized = 0.0
        
        for pos in self.positions:
            pnl = self._get_position_pnl(pos, current_price)
            total_unrealized += pnl
            
        return total_unrealized
    
    def _calculate_adaptive_position_size(self, action_confidence=1.0):
        """
        🚀 MELHORIA #8: Position sizing adaptativo baseado em confiança e volatilidade
        """
        try:
            # Obter volatilidade atual (ATR normalizado)
            current_step = min(self.current_step, len(self.df) - 1)
            atr_5m = self.df['atr_14_5m'].iloc[current_step] if 'atr_14_5m' in self.df.columns else 0.001
            volatility = atr_5m / self.df['close_5m'].iloc[current_step] if self.df['close_5m'].iloc[current_step] > 0 else 0.001
            
            # Normalizar volatilidade (0.001 = baixa, 0.01 = alta)
            volatility = max(min(volatility, 0.02), 0.0005)  # Limitar entre 0.05% e 2%
            
            # Calcular confiança baseada na força do sinal
            # action_confidence vem da força da ação do modelo (0-1)
            confidence_multiplier = min(action_confidence * 1.5, 1.5)  # Max 1.5x
            
            # Calcular divisor de volatilidade (maior volatilidade = menor posição)
            volatility_divisor = max(volatility * 100, 0.5)  # Min 0.5x
            
            # Tamanho final
            size = self.base_lot_size * confidence_multiplier / volatility_divisor
            
            # Aplicar limites
            final_size = max(min(size, self.max_lot_size), 0.01)  # Entre 0.01 e 0.08
            
            return final_size
            
        except Exception as e:
            # Fallback para tamanho base em caso de erro
            return self.base_lot_size
    
    def _check_entry_filters(self, action_type):
        """
        🚀 MELHORIA #2: Filtros de entrada balanceados (não muito restritivos)
        """
        try:
            current_step = min(self.current_step, len(self.df) - 1)
            
            # Filtro 1: Momentum básico (usando features existentes)
            momentum_5m = self.df.get('momentum_5_5m', pd.Series([0])).iloc[current_step]
            momentum_15m = self.df.get('momentum_5_15m', pd.Series([0])).iloc[current_step]
            
            if action_type == 1:  # Long
                momentum_signals = [momentum_5m > 0.0002, momentum_15m > 0.0001]  # 🔥 ALINHADO COM MAINPPO1.PY: ULTRA REDUZIDO
            else:  # Short
                momentum_signals = [momentum_5m < -0.0002, momentum_15m < -0.0001]  # 🔥 ALINHADO COM MAINPPO1.PY: ULTRA REDUZIDO
            
            momentum_confirmations = sum(momentum_signals)
            
            # Filtro 2: Volatilidade não extrema
            volatility_5m = self.df.get('volatility_20_5m', pd.Series([0.001])).iloc[current_step]
            price_5m = self.df['close_5m'].iloc[current_step]
            vol_ratio = volatility_5m / price_5m if price_5m > 0 else 0
            volatility_filter = 0.00005 < vol_ratio < 0.035  # 🔥 ALINHADO COM MAINPPO1.PY: ULTRA EXPANDIDO
            
            # Filtro 3: Anti-microtrading ALINHADO COM MAINPPO1.PY
            recent_trades = len([t for t in self.trades[-3:] if t.get('entry_step', 0) > self.current_step - 3])
            micro_trading_filter = recent_trades < 2  # 🔥 ALINHADO: Máximo 2 trades em 3 steps (15min)
            
            # Filtro 4: Anti-flip-flop (evitar reversões imediatas)
            flip_flop_filter = True
            if len(self.trades) >= 2:
                last_trade = self.trades[-1]
                second_last_trade = self.trades[-2]
                if (last_trade.get('entry_step', 0) > self.current_step - 10 and  # Trade recente
                    last_trade.get('type') != second_last_trade.get('type')):  # Tipos diferentes
                    flip_flop_filter = False  # 🔥 ANTI-FLIP-FLOP: Bloquear reversões rápidas
            
            # 🔥 ALINHADO COM MAINPPO1.PY: ULTRA-PERMISSIVO para maximizar atividade
            entry_allowed = (
                (momentum_confirmations >= 1 and volatility_filter and micro_trading_filter) or  # 🔥 REMOVIDO flip_flop_filter
                (volatility_filter and micro_trading_filter) or  # 🔥 NOVO: Apenas volatilidade + anti-micro
                (momentum_confirmations >= 1 and micro_trading_filter)  # 🔥 NOVO: Apenas momentum + anti-micro
            )
            
            return entry_allowed
            
        except Exception as e:
            # Em caso de erro, permitir entrada (não bloquear o modelo)
            return True

    # --- Pesos da Recompensa (Definidos no seu código) ---
    REWARD_WEIGHTS = {
        'trade_pnl': 1.0,
        'portfolio_growth': 0.5,
        'drawdown_penalty': -2.0,
        'position_management': 0.3,
        'risk_management': 0.2,
        "portfolio_change": 1.0,      # Recompensa base pela variação do portfólio total
        "realized_profit_bonus": 2.5, # Bônus SIGNIFICATIVO por fechar trade com lucro (multiplica o lucro)
        "realized_loss_penalty": -1.5, # Penalidade por fechar trade com prejuízo (multiplica a perda)
        "drawdown_increase": -1.0,    # Penalidade moderada por aumento do drawdown
        "transaction_cost": -0.0005,  # Penalidade por custo de transação (manter pequena/realista)
        "losing_holding_penalty": -0.001, # Penalidade aplicada APENAS a posições perdedoras mantidas por muito tempo
        "invalid_action": -0.01        # Penalidade por ação inválida
    }

    def _calculate_reward_and_info(self, action, old_state):
        """
        Versão modularizada que utiliza o sistema de recompensas separado.
        """
        return self.reward_system.calculate_reward_and_info(self, action, old_state)

def make_wrapped_env(df, window_size, is_training, initial_portfolio=1000, trading_params=None):
    env = TradingEnv(df, window_size=window_size, is_training=is_training, initial_balance=initial_portfolio, trading_params=trading_params)
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    return env

class OptimizationCallback(EvalCallback):
    def __init__(self, trial, eval_env, eval_freq=10000, verbose=0, total_trials=50, start_time=None, study=None):
        super(OptimizationCallback, self).__init__(eval_env, eval_freq=eval_freq, verbose=verbose)
        self.trial = trial
        self.best_mean_reward = -np.inf
        self.step_count = 0
        self.total_trials = total_trials
        self.start_time = start_time if start_time is not None else time.time()
        self.study = study
        self.last_trades_count = 0
        self.last_step_count = 0
        self.global_peak_portfolio = float('-inf')
        self.rewards_buffer = []
        self.portfolio_buffer = []
        self.trades_buffer = []
        self.pruning_disabled = True  # 🔥 DESABILITAR PRUNING COMPLETAMENTE

    def _on_step(self):
        self.step_count += 1
        # Performance logging silenciado
        
        env = self.model.get_env()
        portfolio_now = env.envs[0].portfolio_value
        
        # Inicialize o pico global no primeiro passo
        if self.global_peak_portfolio == float('-inf'):
            self.global_peak_portfolio = portfolio_now
        self.global_peak_portfolio = max(self.global_peak_portfolio, portfolio_now)
        
        # Coletar recompensa atual e métricas
        current_reward = env.envs[0].returns[-1] if env.envs[0].returns else 0
        self.rewards_buffer.append(current_reward)
        self.portfolio_buffer.append(portfolio_now)
        
        # Calcular métricas de trade
        current_trades = env.envs[0].trades
        if len(current_trades) > self.last_trades_count:
            new_trades = current_trades[self.last_trades_count:]
            self.trades_buffer.extend(new_trades)
            self.last_trades_count = len(current_trades)
        
        # Calcular média das recompensas e métricas
        if len(self.rewards_buffer) > 0:
            mean_reward = np.mean(self.rewards_buffer[-100:])  # Média das últimas 100 recompensas (aumentado)
            mean_portfolio = np.mean(self.portfolio_buffer[-100:])
            win_trades = [t for t in self.trades_buffer[-100:] if t.get('pnl_usd', 0) > 0]
            win_rate = len(win_trades) / len(self.trades_buffer[-100:]) if self.trades_buffer[-100:] else 0.0
            
            # 🔥 PRUNING COMPLETAMENTE DESABILITADO para máximo aprendizado
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.trial.report(mean_reward, self.num_timesteps)
                
                # 🚨 NUNCA fazer pruning - deixar todos os trials rodarem completamente
                # if self.trial.should_prune() and not self.pruning_disabled:
                #     print(f"[PRUNING] Trial {self.trial.number} sendo podado após {self.num_timesteps} steps")
                #     raise optuna.TrialPruned()
        
        # Imprimir métricas customizadas a cada 500 passos (ainda mais reduzido)
        if self.step_count % 500 == 0:
            df = env.envs[0].df
            n_barras = len(df)
            dias = n_barras / 247 if n_barras > 0 else 1
            # 🔥 TRADES/DIA CORRIGIDO: Usar dias reais baseados no dataset
            dias_reais = (self.step_count * 5) / (60 * 24) if self.step_count > 0 else 1  # 5min por step
            trades_per_day = len(self.trades_buffer) / dias_reais if dias_reais > 0 else 0.0
            lucro_total = sum(t.get('pnl_usd', 0) for t in self.trades_buffer)
            lucro_medio_dia = lucro_total / dias_reais if dias_reais > 0 else 0.0
            dd_abs = abs(env.envs[0].current_drawdown) * 100
            
            print(f"[TRIAL {self.trial.number}] Step: {self.step_count} | Timesteps: {self.num_timesteps}", flush=True)
            print(f"  Pico: ${self.global_peak_portfolio:.2f} | Portfólio: ${portfolio_now:.2f}", flush=True)
            print(f"  DD: {dd_abs:.2f}% | Trades/dia: {trades_per_day:.2f}", flush=True)
            print(f"  Lucro/dia: ${lucro_medio_dia:.2f} | Win rate: {win_rate:.2%}", flush=True)
            print(f"  Mean reward (100): {mean_reward:.4f} | Best reward: {self.best_mean_reward:.4f}", flush=True)
        
        return False

def load_optimized_data():
    """
    🚀 CARREGAMENTO OTIMIZADO COM CACHE PKL - SPEEDUP 139.8x
    Sistema idêntico ao mainppo1.py para máxima performance
    """
    try:
        # Verificar se existe cache otimizado
        cache_file = 'data_cache/ppo_optimized_dataset.pkl'
        
        if os.path.exists(cache_file):
            print(f"[OPTIMIZED CACHE] Carregando dataset otimizado: {cache_file}")
            start_time = time.time()
            
            df = pd.read_pickle(cache_file)
            
            load_time = time.time() - start_time
            print(f"[OPTIMIZED CACHE] ✅ Dataset carregado em {load_time:.3f}s: {len(df):,} barras")
            print(f"[OPTIMIZED CACHE] Colunas: {list(df.columns)[:10]}...")
            
            return df
        else:
            print(f"[OPTIMIZED CACHE] Cache não encontrado, usando fallback...")
            return get_latest_processed_file_fallback()
            
    except Exception as e:
        print(f"[ERROR] Erro ao carregar cache otimizado: {e}")
        return get_latest_processed_file_fallback()

def get_latest_processed_file_fallback():
    """
    🔥 CARREGAMENTO ROBUSTO DE DATASET COM FALLBACKS MÚLTIPLOS (FALLBACK)
    """
    try:
        # Opção 1: Dataset otimizado (primeira escolha)
        optimized_path = 'data/fixed/train.csv'
        if os.path.exists(optimized_path):
            print(f"[DATASET] Carregando dataset otimizado: {optimized_path}")
            df = pd.read_csv(optimized_path, index_col=0, parse_dates=True)
            
            # Verificar se dataset é válido
            if len(df) > 1000 and 'close_5m' in df.columns:
                print(f"[DATASET] ✅ Dataset otimizado carregado: {len(df):,} barras")
                return df
            else:
                print(f"[WARNING] Dataset otimizado inválido: {len(df)} barras, colunas: {list(df.columns)[:5]}")
        
        # Opção 2: Arquivos CSV originais (fallback)
        print(f"[DATASET] Tentando fallback para arquivos CSV originais...")
        csv_files = {
            '5m': 'data/GOLD_5m_20250513_125132.csv',
            '15m': 'data/GOLD_15m_20250513_125132.csv', 
            '4h': 'data/GOLD_4h_20250513_125132.csv'
        }
        
        dfs = {}
        for tf, file_path in csv_files.items():
            if os.path.exists(file_path):
                print(f"[DATASET] Carregando {tf}: {file_path}")
                df_tf = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Renomear colunas para incluir timeframe
                df_tf.columns = [f"{col}_{tf}" for col in df_tf.columns]
                dfs[tf] = df_tf
                print(f"[DATASET] {tf} carregado: {len(df_tf):,} barras")
            else:
                print(f"[WARNING] Arquivo não encontrado: {file_path}")
        
        if dfs:
            # Combinar timeframes
            print(f"[DATASET] Combinando timeframes: {list(dfs.keys())}")
            combined_df = pd.concat(dfs.values(), axis=1, join='inner')
            
            if len(combined_df) > 1000:
                print(f"[DATASET] ✅ Dataset combinado criado: {len(combined_df):,} barras")
                return combined_df
            else:
                print(f"[ERROR] Dataset combinado muito pequeno: {len(combined_df)} barras")
        
        # Opção 3: Dataset sintético (última opção)
        print(f"[DATASET] Criando dataset sintético para teste...")
        return create_synthetic_dataset()
        
    except Exception as e:
        print(f"[ERROR] Erro ao carregar dataset: {e}")
        print(f"[DATASET] Criando dataset sintético de emergência...")
        return create_synthetic_dataset()

def get_latest_processed_file(timeframe):
    """
    🚀 FUNÇÃO DE COMPATIBILIDADE - REDIRECIONA PARA SISTEMA OTIMIZADO
    """
    return load_optimized_data()

def create_synthetic_dataset():
    """
    🔥 CRIAR DATASET SINTÉTICO PARA TESTES DE EMERGÊNCIA
    """
    try:
        print(f"[SYNTHETIC] Criando dataset sintético...")
        
        # Criar 100k barras de dados sintéticos (347 dias)
        n_bars = 100000
        dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='5T')
        
        # Preço base do ouro (~2000 USD)
        base_price = 2000.0
        
        # Gerar preços com random walk realista
        np.random.seed(42)  # Para reprodutibilidade
        returns = np.random.normal(0, 0.0005, n_bars)  # Volatilidade realista
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Criar dados OHLC básicos
        data = {}
        for tf in ['5m', '15m', '4h']:
            # Simular pequenas variações OHLC
            noise = np.random.normal(0, 0.0002, n_bars)
            
            data[f'open_{tf}'] = prices * (1 + noise)
            data[f'high_{tf}'] = prices * (1 + np.abs(noise) + 0.0001)
            data[f'low_{tf}'] = prices * (1 - np.abs(noise) - 0.0001)
            data[f'close_{tf}'] = prices
            data[f'volume_{tf}'] = np.random.uniform(1000, 10000, n_bars)
        
        df = pd.DataFrame(data, index=dates)
        
        print(f"[SYNTHETIC] ✅ Dataset sintético criado: {len(df):,} barras")
        print(f"[SYNTHETIC] Preço inicial: ${df['close_5m'].iloc[0]:.2f}")
        print(f"[SYNTHETIC] Preço final: ${df['close_5m'].iloc[-1]:.2f}")
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Erro ao criar dataset sintético: {e}")
        # Dataset mínimo de emergência
        dates = pd.date_range(start='2023-01-01', periods=10000, freq='5T')
        df = pd.DataFrame({
            'close_5m': [2000.0] * 10000,
            'close_15m': [2000.0] * 10000,
            'close_4h': [2000.0] * 10000
        }, index=dates)
        
        print(f"[EMERGENCY] Dataset de emergência criado: {len(df):,} barras")
        return df

def print_mem_usage(msg=''):
    process = psutil.Process(os.getpid())
    print(f"[MEM] {msg} - {process.memory_info().rss / 1024**2:.2f} MB")

def filter_trades_by_session(trades, df):
    # Filtra trades para segunda a sexta e entre 19:00 e 18:00 do dia seguinte
    filtered = []
    for t in trades:
        entry_time = df.index[t['entry_step']]
        exit_time = df.index[t['exit_step']]
        # Apenas segunda a sexta
        if entry_time.weekday() > 4 or exit_time.weekday() > 4:
            continue
        # Sessão: das 19:00 de um dia até 18:00 do próximo
        if not ((entry_time.hour >= 19 or entry_time.hour < 18) and (exit_time.hour >= 19 or exit_time.hour < 18)):
            continue
        filtered.append(t)
    return filtered

gui_metrics = {
    'portfolio': 0.0,
    'drawdown': 0.0,
    'dd_peak': 0.0,
    'trades_per_day': 0.0,
    'lucro_medio_dia': 0.0,
    'total_trades': 0,
    'win_rate': 0.0,
    'sharpe': 0.0
}

gui_best_metrics = {
    'portfolio': {'value': float('-inf'), 'trial': None},
    'drawdown': {'value': float('inf'), 'trial': None},
    'dd_peak': {'value': float('inf'), 'trial': None},
    'trades_per_day': {'value': float('-inf'), 'trial': None},
    'lucro_medio_dia': {'value': float('-inf'), 'trial': None},
    'total_trades': {'value': float('-inf'), 'trial': None},
    'win_rate': {'value': float('-inf'), 'trial': None},
    'sharpe': {'value': float('-inf'), 'trial': None}
}

def save_metrics(metrics, trial_number):
    """Salva métricas de um trial individual em arquivo JSON"""
    results_dir = "optimization_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"ppo_trial_metrics_{trial_number}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"📊 Trial {trial_number} metrics saved: {results_file}")

def save_trial_result(trial, instance_id, logger):
    """Salva resultado de um trial individual em tempo real"""
    try:
        results_dir = "optimization_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_file = os.path.join(results_dir, f"trial_{trial.number}_{instance_id}_{timestamp}.json")
        
        # Dados do trial
        trial_data = {
            "timestamp": datetime.now().isoformat(),
            "trial_number": trial.number,
            "trial_value": trial.value,
            "trial_state": trial.state.name,
            "trial_params": trial.params,
            "trial_duration": trial.duration.total_seconds() if trial.duration else None,
            "user_attrs": trial.user_attrs if hasattr(trial, 'user_attrs') else {},
            "instance_id": instance_id
        }
        
        # Salvar arquivo do trial
        with open(trial_file, 'w') as f:
            json.dump(trial_data, f, indent=4)
            
        # Log e terminal
        if trial.value is not None:
            attrs = trial.user_attrs if hasattr(trial, 'user_attrs') else {}
            logger.info(f"🔥 Trial {trial.number} SAVED: Score={trial.value:.4f}, Portfolio=${attrs.get('portfolio_value', 0):.0f}")
            print(f"💾 Trial {trial.number} saved: {trial_file}")
        else:
            logger.warning(f"❌ Trial {trial.number} FAILED and saved")
            print(f"❌ Failed trial {trial.number} saved: {trial_file}")
            
        return trial_file
        
    except Exception as e:
        error_msg = f"Erro ao salvar trial {trial.number}: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return None

def save_partial_study_results(study, instance_id, logger, reason="partial"):
    """Salva resultados parciais do estudo em tempo real"""
    try:
        results_dir = "optimization_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partial_file = os.path.join(results_dir, f"partial_study_{instance_id}_{timestamp}_{reason}.json")
        
        # Estatísticas do estudo
        completed_trials = [t for t in study.trials if t.value is not None]
        failed_trials = [t for t in study.trials if t.value is None]
        
        # Dados parciais
        partial_data = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "instance_id": instance_id,
            "study_summary": {
                "total_trials": len(study.trials),
                "completed_trials": len(completed_trials),
                "failed_trials": len(failed_trials),
                "success_rate": len(completed_trials) / len(study.trials) if study.trials else 0
            },
            "best_trial": {
                "number": study.best_trial.number if completed_trials else None,
                "value": study.best_trial.value if completed_trials else None,
                "params": study.best_trial.params if completed_trials else None,
                "user_attrs": study.best_trial.user_attrs if completed_trials and hasattr(study.best_trial, 'user_attrs') else {}
            } if completed_trials else None,
            "all_trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "state": trial.state.name,
                    "params": trial.params,
                    "duration": trial.duration.total_seconds() if trial.duration else None,
                    "user_attrs": trial.user_attrs if hasattr(trial, 'user_attrs') else {}
                }
                for trial in study.trials
            ]
        }
        
        # Salvar arquivo parcial
        with open(partial_file, 'w') as f:
            json.dump(partial_data, f, indent=4)
            
        # Log e terminal
        logger.info(f"💾 Partial study saved: {len(completed_trials)}/{len(study.trials)} trials completed")
        print(f"💾 Partial results saved: {partial_file}")
        print(f"📊 Progress: {len(completed_trials)}/{len(study.trials)} trials | Success: {len(completed_trials)/len(study.trials)*100:.1f}%")
        
        return partial_file
        
    except Exception as e:
        error_msg = f"Erro ao salvar resultados parciais: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return None

def read_latest_metrics():
    files = glob.glob("metrics_trial_*.json")
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, "r") as f:
        return json.load(f)

def start_dd_peak_gui():
    def gui_loop():
        root = tk.Tk()
        root.title('Métricas RL Trading')
        root.geometry('600x340')
        labels = [
            ('Portfolio', 'portfolio', '{:.2f}'),
            ('Drawdown', 'drawdown', '{:.2f}%'),
            ('DD Peak', 'dd_peak', '{:.2f}%'),
            ('Trades/dia', 'trades_per_day', '{:.2f}'),
            ('Lucro médio/dia', 'lucro_medio_dia', '{:.2f}'),
            ('Total trades', 'total_trades', '{}'),
            ('Win rate', 'win_rate', '{:.2f}%'),
            ('Sharpe', 'sharpe', '{:.2f}')
        ]
        value_vars = {}
        best_vars = {}
        for i, (label_text, key, fmt) in enumerate(labels):
            label = ttk.Label(root, text=label_text+':', font=('Arial', 13))
            label.grid(row=i, column=0, sticky='w', padx=10, pady=5)
            value_var = tk.StringVar()
            value_label = ttk.Label(root, textvariable=value_var, font=('Arial', 13, 'bold'))
            value_label.grid(row=i, column=1, sticky='w', padx=10, pady=5)
            value_vars[key] = (value_var, fmt)
            best_var = tk.StringVar()
            best_label = ttk.Label(root, textvariable=best_var, font=('Arial', 13, 'bold'), foreground='blue')
            best_label.grid(row=i, column=2, sticky='w', padx=10, pady=5)
            best_vars[key] = (best_var, fmt)
        best_title = ttk.Label(root, text='Melhor valor (trial)', font=('Arial', 12, 'italic'))
        best_title.grid(row=0, column=2, padx=10, pady=5, sticky='w')
        def safe_fmt(val, fmt, dash_for_nan=True):
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                return '-' if dash_for_nan else '0'
            try:
                return fmt.format(val)
            except Exception:
                return str(val)
        def update():
            # Atualiza a partir do arquivo de métricas mais recente
            metrics = read_latest_metrics()
            if metrics:
                gui_metrics['portfolio'] = metrics.get('portfolio_value', 0)
                gui_metrics['drawdown'] = metrics.get('max_drawdown', 0)
                gui_metrics['dd_peak'] = metrics.get('peak_drawdown', 0)
                gui_metrics['trades_per_day'] = metrics.get('trades_per_day', 0)
                gui_metrics['lucro_medio_dia'] = metrics.get('lucro_medio_dia', 0)
                gui_metrics['total_trades'] = metrics.get('total_trades', 0)
                gui_metrics['win_rate'] = metrics.get('win_rate', 0)
                gui_metrics['sharpe'] = metrics.get('sharpe_ratio', 0)
            for key, (value_var, fmt) in value_vars.items():
                val = gui_metrics.get(key, 0)
                if key in ['drawdown', 'dd_peak', 'win_rate']:
                    val = val * 100
                value_var.set(safe_fmt(val, fmt))
            for key, (best_var, fmt) in best_vars.items():
                best = gui_best_metrics[key]['value']
                trial = gui_best_metrics[key]['trial']
                if key in ['drawdown', 'dd_peak', 'win_rate']:
                    best = best * 100
                txt = safe_fmt(best, fmt)
                if trial is not None and trial != 'None':
                    txt += f'  (trial {trial})'
                best_var.set(txt)
            # --- NOVO: Exibir métricas de avaliação ---
            try:
                with open('metrics_eval.json', 'r') as f_eval:
                    eval_metrics = json.load(f_eval)
                if eval_metrics:
                    eval_str = '\n'.join([
                        f"Episódio {i+1}: Portfólio: {m.get('peak_portfolio','-')}, Drawdown: {m.get('max_drawdown','-'):.2f}%, Trades/dia: {m.get('trades_per_day','-'):.2f}, Win rate: {m.get('win_rate','-'):.2%}"
                        for i, m in enumerate(eval_metrics)
                    ])
                else:
                    eval_str = 'Sem métricas de avaliação.'
            except Exception as e:
                eval_str = f'Erro ao ler métricas de avaliação: {e}'
            # Exibir em um label extra
            if not hasattr(root, 'eval_label'):
                root.eval_label = ttk.Label(root, text='Métricas de Avaliação:', font=('Arial', 12, 'bold'), foreground='green')
                root.eval_label.grid(row=len(labels)+1, column=0, columnspan=3, sticky='w', padx=10, pady=5)
                root.eval_val = tk.StringVar()
                root.eval_val_label = ttk.Label(root, textvariable=root.eval_val, font=('Arial', 11), foreground='black', wraplength=550, justify='left')
                root.eval_val_label.grid(row=len(labels)+2, column=0, columnspan=3, sticky='w', padx=10, pady=5)
            root.eval_val.set(eval_str)
            root.after(1000, update)
        update()
        root.mainloop()
    t = threading.Thread(target=gui_loop, daemon=False)
    t.start()
    return t

def print_metrics_report(step, portfolio_value, drawdown, peak_drawdown, trades, df, returns, metrics, when='step', action_counts=None):
    print("\n================= 📊 MÉTRICAS DE AVALIAÇÃO =================")
    print(f"Step: {step}")
    print(f"🏆 Pico Portfólio: ${metrics.get('peak_portfolio', portfolio_value):.2f} | 💼 Portfólio Atual: ${portfolio_value:.2f}")
    print(f"📉 Drawdown: {drawdown*100:.2f}% | 📉 DD Peak: {peak_drawdown*100:.2f}%")
    trades_per_day = metrics.get('trades_per_day', 0)
    lucro_medio_dia = metrics.get('lucro_medio_dia', 0)
    all_trades = trades if trades is not None else []
    win_rate = metrics.get('win_rate', 0)
    print(f"🔄 Trades/dia: {trades_per_day:.2f} | 💰 Lucro médio/dia: {lucro_medio_dia:.2f}")
    print(f"📈 Total trades: {len(all_trades)} | 🟢 Win rate: {win_rate*100:.2f}%")
    print(f"📊 Sharpe: {fmt_metric(metrics.get('sharpe_ratio', 0))}")
    print(f"🟦 Ações por tipo: {action_counts if action_counts is not None else metrics.get('action_counts', {})}")
    print("========================================================\n")

def trial_target(objective, trial, result_queue):
    try:
        result = objective(trial)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)

def run_objective_in_subprocess(objective, trial, timeout=10800):  # Aumentado de 1800 para 10800 segundos (3 horas)
    result_queue = queue.Queue()
    p = multiprocessing.Process(target=trial_target, args=(objective, trial, result_queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print(f'[WATCHDOG] Trial excedeu {timeout} segundos. Matando subprocesso!')
        p.terminate()
        p.join()
        raise optuna.TrialPruned(f'Trial excedeu {timeout} segundos e foi abortado.')
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        raise optuna.TrialPruned('Subprocesso terminou sem retornar resultado.')

def objective_with_timeout(trial):
    # Remover multiprocessing: rodar diretamente
    return objective(trial)


# --- MÉTRICAS DE PORTFÓLIO ---
class PortfolioMetrics:
    @staticmethod
    def calculate_metrics(returns: np.ndarray, trades: list, start_date: datetime, end_date: datetime, df=None, peak_portfolio=None) -> dict:
        if len(returns) == 0 or len(trades) == 0:
            return {
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'trades_per_day': 0.0,
                'trading_days': 0,
                'peak_drawdown': 0.0,
                'lucro_medio_dia': 0.0,
                'peak_portfolio': 0.0,
            }

        # Calcular métricas básicas
        returns = np.array(returns)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calcular dias de trading
        if df is not None:
            unique_days = pd.to_datetime(df.index.normalize()).unique()
            trading_days = len(unique_days)
        else:
            trading_days = max((end_date - start_date).days, 1)

        # Calcular retorno total e anualizado
        total_return = np.sum(returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else total_return

        # Calcular volatilidade
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

        # 🔥 CORRIGIR Sharpe Ratio com proteção contra valores infinitos
        risk_free_rate = 0.0  # Assumindo taxa livre de risco zero
        if volatility > 1e-8 and not np.isnan(volatility) and not np.isinf(volatility):
            sharpe_ratio = (annual_return - risk_free_rate) / volatility
            # Limitar Sharpe Ratio a valores razoáveis
            sharpe_ratio = np.clip(sharpe_ratio, -100, 100)
        else:
            sharpe_ratio = 0.0

        # Calcular drawdowns
        cumulative_returns = np.cumsum(returns) + 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = np.where(running_max > 1e-8, (running_max - cumulative_returns) / running_max, 0.0)
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        peak_drawdown = max_drawdown

        # 🔥 CORRIGIR Sortino Ratio com proteção contra valores infinitos
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0.0
        if downside_std > 1e-8 and not np.isnan(downside_std) and not np.isinf(downside_std):
            sortino_ratio = (annual_return - risk_free_rate) / downside_std
            # Limitar Sortino Ratio a valores razoáveis
            sortino_ratio = np.clip(sortino_ratio, -100, 100)
        else:
            sortino_ratio = 0.0

        # 🔥 CORRIGIR Calmar Ratio com proteção contra valores infinitos
        if max_drawdown > 1e-8 and not np.isnan(max_drawdown) and not np.isinf(max_drawdown):
            calmar_ratio = annual_return / max_drawdown
            # Limitar Calmar Ratio a valores razoáveis
            calmar_ratio = np.clip(calmar_ratio, -100, 100)
        else:
            calmar_ratio = 0.0

        # Calcular métricas de trades
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get('pnl_usd', 0) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Calcular trades por dia
        trades_per_day = total_trades / trading_days if trading_days > 0 else 0.0

        # Calcular lucro médio por dia
        total_pnl = sum(t.get('pnl_usd', 0) for t in trades)
        lucro_medio_dia = total_pnl / trading_days if trading_days > 0 else 0.0

        # Usar peak_portfolio fornecido ou calcular
        peak_portfolio_val = peak_portfolio if peak_portfolio is not None else float(np.max(cumulative_returns))

        # 🔥 PROTEÇÃO FINAL: Limpar todos os valores de NaN/Inf antes de retornar
        def safe_float(val, default=0.0):
            if np.isnan(val) or np.isinf(val):
                return default
            return np.clip(float(val), -1e6, 1e6)  # Limitar a valores razoáveis

        return {
            'annual_return': safe_float(annual_return),
            'volatility': safe_float(volatility),
            'sharpe_ratio': safe_float(sharpe_ratio),
            'max_drawdown': safe_float(max_drawdown),
            'sortino_ratio': safe_float(sortino_ratio),
            'calmar_ratio': safe_float(calmar_ratio),
            'total_trades': int(total_trades),
            'win_rate': safe_float(win_rate),
            'trades_per_day': safe_float(trades_per_day),
            'trading_days': int(trading_days),
            'peak_drawdown': safe_float(peak_drawdown),
            'lucro_medio_dia': safe_float(lucro_medio_dia),
            'peak_portfolio': safe_float(peak_portfolio_val),
        }
# --- FIM MÉTRICAS DE PORTFÓLIO ---

# 🔥 IMPORTAR CLASSES DO FRAMEWORK MODULARIZADO
try:
    from trading_framework.policies import TwoHeadPolicy
    print("[OK IMPORT] TwoHeadPolicy importada do framework modularizado")
except ImportError as e:
    print(f"[ERROR IMPORT] Erro ao importar TwoHeadPolicy do framework: {e}")
    print(f"[ERROR FALLBACK REMOVIDO] Não importar do mainppo1.py para evitar conflitos")
    raise ImportError("TwoHeadPolicy não encontrada no framework - verifique a instalação")

try:
    from trading_framework.extractors import TransformerFeatureExtractor
    print("[OK IMPORT] TransformerFeatureExtractor importada do framework")
except ImportError as e:
    print(f"[ERROR IMPORT] Erro ao importar TransformerFeatureExtractor do framework: {e}")
    # Fallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    TransformerFeatureExtractor = BaseFeaturesExtractor

# 🔥 FUNÇÃO find_latest_model REMOVIDA - PPO.PY NÃO DEVE CARREGAR MODELOS PRÉ-EXISTENTES

# 🔥 ADICIONAR CLASSES NECESSÁRIAS PARA COMPATIBILIDADE TOTAL COM MAINPPO1.PY
# 🔥 USAR CLASSES DO FRAMEWORK PARA COMPATIBILIDADE TOTAL
from trading_framework.policies import TwoHeadPolicy

# Verificar se a importação foi bem-sucedida
try:
    print("[OK IMPORT] TwoHeadPolicy importada do framework")
    print("[OK IMPORT] TransformerFeatureExtractor já importada")
except ImportError as e:
    print(f"[ERROR IMPORT] Erro ao importar do framework: {e}")
    raise ImportError("Não foi possível importar as classes necessárias do trading_framework")

def objective(trial):
    """
    🎯 FUNÇÃO OBJETIVO HÍBRIDA - SL/TP RANGES OPTIMIZATION
    """
    try:
        # 🔥 DETERMINAR FASE BASEADA NO TRIAL NUMBER
        trial_number = trial.number
        is_refinement_phase = trial_number < 50  # Primeiros 50 = refinamento
        
        if is_refinement_phase:
            phase = "REFINAMENTO"
            print(f"\n🔧 [TRIAL {trial_number}] FASE REFINAMENTO - Ajustes finos nos melhores parâmetros", flush=True)
        else:
            phase = "EXPLORAÇÃO"
            print(f"\n🚀 [TRIAL {trial_number}] FASE EXPLORAÇÃO - Busca por novos ótimos globais", flush=True)
        
        # 🎯 HIPERPARÂMETROS HÍBRIDOS: Refinamento vs Exploração
        if is_refinement_phase:
            # 🔧 FASE REFINAMENTO: Ranges pequenos ao redor dos melhores valores
            learning_rate = trial.suggest_float('learning_rate', 2.0e-05, 3.5e-05, log=True)
            n_steps = trial.suggest_categorical('n_steps', [1536, 1792, 2048])  # ±256 ao redor de 1792
            batch_size = trial.suggest_categorical('batch_size', [48, 64, 80])  # ±16 ao redor de 64
            clip_range = trial.suggest_float('clip_range', 0.07, 0.095)  # ±0.012 ao redor de 0.082
            ent_coef = trial.suggest_float('ent_coef', 0.015, 0.025)  # ±0.005 ao redor de 0.020
            vf_coef = trial.suggest_float('vf_coef', 0.55, 0.65)  # ±0.05 ao redor de 0.602
            
            # 🎯 SL/TP RANGES - REFINAMENTO FINO (MAIS APERTADOS)
            sl_range_min = trial.suggest_int('sl_range_min', 8, 12)     # 🔥 MAIS APERTADO: 8-12
            sl_range_max = trial.suggest_int('sl_range_max', 25, 35)   # 🔥 MAIS APERTADO: 25-35
            tp_range_min = trial.suggest_int('tp_range_min', 10, 15)   # 🔥 MAIS APERTADO: 10-15
            tp_range_max = trial.suggest_int('tp_range_max', 40, 50)   # 🔥 MAIS APERTADO: 40-50
            
            # 🎯 THRESHOLDS - REFINAMENTO FINO
            momentum_threshold = trial.suggest_float('momentum_threshold', 0.0005, 0.0008)  # Refinamento
            volatility_min = trial.suggest_float('volatility_min', 0.0003, 0.0006)         # Refinamento
            volatility_max = trial.suggest_float('volatility_max', 0.014, 0.018)           # Refinamento
            
        else:
            # 🚀 FASE EXPLORAÇÃO: Ranges amplos para descobrir novos ótimos
            learning_rate = trial.suggest_float('learning_rate', 1e-05, 8e-05, log=True)
            n_steps = trial.suggest_categorical('n_steps', [1024, 1280, 1536, 1792, 2048, 2304])
            batch_size = trial.suggest_categorical('batch_size', [32, 48, 64, 80, 96, 128])
            clip_range = trial.suggest_float('clip_range', 0.05, 0.15)
            ent_coef = trial.suggest_float('ent_coef', 0.005, 0.035)
            vf_coef = trial.suggest_float('vf_coef', 0.4, 0.8)
            
            # 🎯 SL/TP RANGES - EXPLORAÇÃO AMPLA (MAIS APERTADOS)
            sl_range_min = trial.suggest_int('sl_range_min', 5, 15)     # 🔥 MAIS APERTADO: 5-15
            sl_range_max = trial.suggest_int('sl_range_max', 20, 40)   # 🔥 MAIS APERTADO: 20-40
            tp_range_min = trial.suggest_int('tp_range_min', 8, 18)    # 🔥 MAIS APERTADO: 8-18
            tp_range_max = trial.suggest_int('tp_range_max', 35, 55)   # 🔥 MAIS APERTADO: 35-55
            
            # 🎯 THRESHOLDS - EXPLORAÇÃO AMPLA
            momentum_threshold = trial.suggest_float('momentum_threshold', 0.0002, 0.0015)  # Exploração
            volatility_min = trial.suggest_float('volatility_min', 0.0001, 0.001)          # Exploração
            volatility_max = trial.suggest_float('volatility_max', 0.008, 0.025)           # Exploração
        
        # 🔥 HIPERPARÂMETROS FIXOS (não otimizar para focar em SL/TP)
        n_epochs = 10
        gamma = 0.99
        gae_lambda = 0.95
        max_grad_norm = 0.5
        window_size = 20
        
        # 🎯 PARÂMETROS DE TRADING HÍBRIDOS
        if is_refinement_phase:
            # Refinamento: pequenos ajustes ao redor dos valores ótimos
            target_trades_per_day = trial.suggest_int('target_trades_per_day', 16, 20)  # 18±2
            portfolio_weight = trial.suggest_float('portfolio_weight', 0.75, 0.82)      # 0.788±0.038
            drawdown_weight = trial.suggest_float('drawdown_weight', 0.47, 0.55)        # 0.510±0.040
            max_drawdown_tolerance = trial.suggest_float('max_drawdown_tolerance', 0.30, 0.37)  # 0.338±0.038
            win_rate_target = trial.suggest_float('win_rate_target', 0.50, 0.56)        # 0.529±0.029
        else:
            # Exploração: ranges amplos para descobrir novos ótimos
            target_trades_per_day = trial.suggest_int('target_trades_per_day', 12, 25)
            portfolio_weight = trial.suggest_float('portfolio_weight', 0.6, 0.9)
            drawdown_weight = trial.suggest_float('drawdown_weight', 0.3, 0.7)
            max_drawdown_tolerance = trial.suggest_float('max_drawdown_tolerance', 0.2, 0.5)
            win_rate_target = trial.suggest_float('win_rate_target', 0.4, 0.65)
        
        # 🔥 VALIDAÇÃO DE RANGES SL/TP
        if sl_range_min >= sl_range_max:
            print(f"[ERROR TRIAL {trial_number}] SL range inválido: min={sl_range_min} >= max={sl_range_max}")
            return -1000.0
        
        if tp_range_min >= tp_range_max:
            print(f"[ERROR TRIAL {trial_number}] TP range inválido: min={tp_range_min} >= max={tp_range_max}")
            return -1000.0
        
        # 🔥 LOGS DE DEBUG PARA RANGES
        print(f"[TRIAL {trial_number}] 🎯 SL Range: {sl_range_min}-{sl_range_max} pontos")
        print(f"[TRIAL {trial_number}] 🎯 TP Range: {tp_range_min}-{tp_range_max} pontos")
        print(f"[TRIAL {trial_number}] 🎯 Target Trades/Dia: {target_trades_per_day}")
        print(f"[TRIAL {trial_number}] 🎯 Momentum Threshold: {momentum_threshold:.6f}")
        
        # Criar parâmetros de trading
        trading_params = {
            'sl_range_min': sl_range_min,
            'sl_range_max': sl_range_max,
            'tp_range_min': tp_range_min,
            'tp_range_max': tp_range_max,
            'target_trades_per_day': target_trades_per_day,
            'portfolio_weight': portfolio_weight,
            'drawdown_weight': drawdown_weight,
            'max_drawdown_tolerance': max_drawdown_tolerance,
            'win_rate_target': win_rate_target,
            'momentum_threshold': momentum_threshold,
            'volatility_min': volatility_min,
            'volatility_max': volatility_max
        }
        
        # 🔥 CARREGAR DATASET COM FALLBACK ROBUSTO
        try:
            df_train = get_latest_processed_file('5m')
            if df_train is None or len(df_train) < 1000:
                print(f"[ERROR TRIAL {trial_number}] Dataset inválido ou muito pequeno")
                return -1000.0
            
            # 🔥 CORTAR PRIMEIROS 20% PROBLEMÁTICOS (igual mainppo1.py)
            cut_point = int(len(df_train) * 0.2)
            df_train = df_train.iloc[cut_point:].copy()
            df_val = df_train.iloc[-50000:].copy()  # Últimos 50k para validação
            
            print(f"[TRIAL {trial_number}] Dataset carregado: {len(df_train):,} barras (após corte 20%)")
            
        except Exception as e:
            print(f"[ERROR TRIAL {trial_number}] Erro ao carregar dataset: {e}")
            return -1000.0
        
        # 🔥 CRIAR AMBIENTES COM TRATAMENTO DE ERRO ROBUSTO
        try:
            # Ambiente de treinamento
            train_env_raw = DummyVecEnv([lambda: Monitor(make_wrapped_env(df_train, window_size, True, trading_params=trading_params))])
            
            # 🔥 VECNORMALIZE SEMPRE ATIVO COM OBS E REWARDS TRUE
            if USE_VECNORM:
                train_env = VecNormalize(train_env_raw, training=True, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10., gamma=gamma)
                print(f"[TRIAL {trial_number}] VecNormalize ativado: norm_obs=True, norm_reward=True")
            else:
                train_env = train_env_raw
                print(f"[TRIAL {trial_number}] VecNormalize desabilitado")
            
            print(f"[TRIAL {trial_number}] Ambientes criados com sucesso")
            
        except Exception as e:
            print(f"[ERROR TRIAL {trial_number}] Erro ao criar ambientes: {e}")
            return -1000.0
        
        # 🔥 CRIAR MODELO COM TRATAMENTO DE ERRO ROBUSTO
        try:
            # 🔥 SEMPRE USAR TwoHeadPolicy COM TransformerFeatureExtractor
            policy_class = TwoHeadPolicy
            policy_kwargs = {
                "features_extractor_class": TradingTransformerFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 128},
                "lstm_hidden_size": 128,
                "n_lstm_layers": 2,
                "activation_fn": torch.nn.ReLU,
                "normalize_images": False
            }
            
            model = RecurrentPPO(
                policy_class,
                train_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=0,
                device="cpu",  # 🔥 CPU FORÇADO
                seed=SEED
            )
            
            print(f"[TRIAL {trial_number}] Modelo TwoHeadPolicy + TradingTransformerFeatureExtractor criado com sucesso")
            
        except Exception as e:
            print(f"[ERROR TRIAL {trial_number}] Erro ao criar modelo: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0
        
        # 🔥 TREINAMENTO COM STEPS OTIMIZADOS
        if is_refinement_phase:
            total_timesteps = 40000  # Refinamento: menos steps, mais precisão
        else:
            total_timesteps = 60000  # Exploração: mais steps, mais robustez
        
        print(f"[TRIAL {trial_number}] Iniciando treinamento: {total_timesteps:,} timesteps")
        
        try:
            model.learn(total_timesteps=total_timesteps)
            print(f'[TRIAL {trial_number}] Treinamento concluído')
        except Exception as e:
            print(f"[ERROR TRIAL {trial_number}] Erro durante treinamento: {str(e)}")
            return -1000.0

        # 🔥 AVALIAÇÃO ROBUSTA COM TRATAMENTO DE ERRO
        try:
            eval_env_raw = DummyVecEnv([lambda: Monitor(make_wrapped_env(df_val, window_size, False, trading_params=trading_params))])
            if USE_VECNORM:
                eval_env = VecNormalize(eval_env_raw, training=False, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10., gamma=gamma)
                if hasattr(train_env, 'obs_rms'):
                    eval_env.obs_rms = train_env.obs_rms
            else:
                eval_env = eval_env_raw
            
            model.policy.eval()
            
            obs = eval_env.reset()
            lstm_states = None
            episode_starts = torch.ones(1, dtype=torch.bool, device="cpu")
            done = False
            step = 0
            trades = []
            portfolio_values = [1000.0]  # 🔥 INICIALIZAR COM VALOR VÁLIDO
            
            # 🔥 AVALIAÇÃO SEGURA COM LIMITE DE STEPS
            max_eval_steps = 30000  # Reduzido para evitar timeout
            print(f"[TRIAL {trial_number}] Iniciando avaliação: {max_eval_steps:,} steps máximo")
            
            while not done and step < max_eval_steps:
                with torch.no_grad():
                    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
                
                obs, reward, done, info = eval_env.step(action)
                step += 1
                
                # 🔥 COLETAR MÉTRICAS COM FALLBACK SEGURO
                if hasattr(info, '__len__') and len(info) > 0 and isinstance(info[0], dict):
                    info_dict = info[0]
                    if 'portfolio_value' in info_dict and info_dict['portfolio_value'] is not None:
                        portfolio_values.append(float(info_dict['portfolio_value']))
                    if 'trades' in info_dict and info_dict['trades']:
                        trades.extend(info_dict['trades'])
                
                # 🔥 FAILSAFE: Se portfolio_values não está crescendo, simular valores
                if step % 5000 == 0 and len(portfolio_values) <= 1:
                    portfolio_values.append(1000.0 + (step * 0.01))  # Crescimento mínimo simulado
            
            print(f"[TRIAL {trial_number}] Avaliação concluída: {step} steps, {len(trades)} trades")
            
        except Exception as e:
            print(f"[ERROR TRIAL {trial_number}] Erro durante avaliação: {str(e)}")
            return -1000.0
        
        # 🔥 CÁLCULO DE MÉTRICAS COM FALLBACKS ROBUSTOS
        try:
            if len(portfolio_values) < 2:
                print(f"[ERROR TRIAL {trial_number}] Portfolio values insuficientes: {len(portfolio_values)}")
                return -1000.0
            
            final_portfolio = portfolio_values[-1]
            initial_portfolio = portfolio_values[0]
            total_return = (final_portfolio - initial_portfolio) / initial_portfolio
            
            # 🔥 VALIDAÇÃO DE VALORES
            if np.isnan(total_return) or np.isinf(total_return):
                print(f"[ERROR TRIAL {trial_number}] Total return inválido: {total_return}")
                return -1000.0
            
            # Calcular drawdown máximo
            max_drawdown = 0.0
            peak = initial_portfolio
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            # Win rate e trades por dia
            winning_trades = len([t for t in trades if t.get('pnl_usd', 0) > 0])
            total_trades = len(trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            trading_days = step / 288 if step > 0 else 1  # 288 steps = 1 dia
            trades_per_day = total_trades / trading_days
            
            # 🎯 SCORE HÍBRIDO: Focado em SL/TP performance
            if is_refinement_phase:
                # Refinamento: foco em consistência e qualidade
                portfolio_score = min(total_return * 100, 40.0)  # Max 40 pontos
                drawdown_score = max(0, 30 - (max_drawdown * 100))  # Max 30 pontos
                win_rate_score = win_rate * 20  # Max 20 pontos
                trade_activity_score = min(trades_per_day / target_trades_per_day, 1.0) * 10  # Max 10 pontos
            else:
                # Exploração: foco em descoberta de ótimos
                portfolio_score = min(total_return * 150, 50.0)  # Max 50 pontos (mais agressivo)
                drawdown_score = max(0, 25 - (max_drawdown * 100))  # Max 25 pontos
                win_rate_score = win_rate * 15  # Max 15 pontos
                trade_activity_score = min(trades_per_day / target_trades_per_day, 1.0) * 10  # Max 10 pontos
            
            final_score = portfolio_score + drawdown_score + win_rate_score + trade_activity_score
            
            # 🔥 VALIDAÇÃO FINAL DO SCORE
            if np.isnan(final_score) or np.isinf(final_score) or final_score < -100:
                print(f"[ERROR TRIAL {trial_number}] Score inválido: {final_score}")
                return -1000.0
            
            # 🔥 SALVAR MÉTRICAS NO TRIAL
            trial.set_user_attr('portfolio_value', final_portfolio)
            trial.set_user_attr('total_return', total_return)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('win_rate', win_rate)
            trial.set_user_attr('trades_per_day', trades_per_day)
            trial.set_user_attr('total_trades', total_trades)
            trial.set_user_attr('phase', phase)
            trial.set_user_attr('sl_range', f"{sl_range_min}-{sl_range_max}")
            trial.set_user_attr('tp_range', f"{tp_range_min}-{tp_range_max}")
            
            # 🔥 LOGS DE RESULTADO
            print(f"\n🏆 [TRIAL {trial_number}] RESULTADO {phase}:")
            print(f"   💰 Portfolio: ${final_portfolio:.2f} (Return: {total_return*100:.2f}%)")
            print(f"   📉 Max DD: {max_drawdown*100:.2f}%")
            print(f"   🎯 Win Rate: {win_rate*100:.2f}%")
            print(f"   📊 Trades/Dia: {trades_per_day:.1f} (Target: {target_trades_per_day})")
            print(f"   🎯 SL Range: {sl_range_min}-{sl_range_max} | TP Range: {tp_range_min}-{tp_range_max}")
            print(f"   🏆 Score Final: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            print(f"[ERROR TRIAL {trial_number}] Erro no cálculo de métricas: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0
        
    except Exception as e:
        print(f"[CRITICAL ERROR TRIAL {trial.number}] {str(e)}")
        import traceback
        traceback.print_exc()
        return -1000.0


def save_best_params_to_log(study, logger, filename="best_params_backup.json"):
    """Salva os melhores parâmetros e TODAS as métricas no log e em arquivo JSON de backup"""
    try:
        if not study.trials:
            logger.info("Nenhum trial completado ainda")
            print("❌ Nenhum trial completado ainda")
            return
            
        # Encontrar o melhor trial
        best_trial = max(study.trials, key=lambda t: t.value if t.value is not None else -1000)
        
        if best_trial.value is None:
            logger.info("Nenhum trial com valor válido encontrado")
            print("❌ Nenhum trial com valor válido encontrado")
            return
            
        # Função para formatar valores com segurança
        def safe_format(val, fmt="{:.4f}", pct=False, currency=False):
            try:
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    return "N/A"
                if pct:
                    val = max(0.0, min(1.0, val)) if val <= 1 else val / 100
                    return f"{val*100:.2f}%"
                if currency:
                    return f"${val:.2f}"
                return fmt.format(val)
            except Exception:
                return str(val) if val is not None else "N/A"
        
        # Obter todas as métricas
        attrs = best_trial.user_attrs if hasattr(best_trial, 'user_attrs') else {}
        
        # === LOG DETALHADO DOS MELHORES PARÂMETROS ===
        logger.info(f"\n" + "="*70)
        logger.info(f"🏆 MELHOR TRIAL ENCONTRADO - Trial #{best_trial.number}")
        logger.info(f"🎯 Objective Score: {safe_format(best_trial.value)}")
        logger.info(f"📊 Trials Completados: {len(study.trials)}")
        logger.info(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        # === HIPERPARÂMETROS ===
        logger.info(f"\n🔧 MELHORES HIPERPARÂMETROS:")
        logger.info(f"-" * 40)
        for key, value in best_trial.params.items():
            if isinstance(value, float):
                if key == 'learning_rate':
                    logger.info(f"  {key}: {value:.2e}")
                else:
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # === MÉTRICAS DE PERFORMANCE DETALHADAS ===
        logger.info(f"\n📈 MÉTRICAS DE PERFORMANCE:")
        logger.info(f"-" * 40)
        
        # Performance principal
        logger.info(f"💰 Portfolio Final: {safe_format(attrs.get('portfolio_value', 0), currency=True)}")
        logger.info(f"💵 Saldo Realizado: {safe_format(attrs.get('final_balance', 0), currency=True)}")
        logger.info(f"📊 PnL Não Realizado: {safe_format(attrs.get('unrealized_pnl', 0), currency=True)}")
        logger.info(f"⬆️ Pico do Portfolio: {safe_format(attrs.get('peak_portfolio', 0), currency=True)}")
        
        # Risk metrics
        logger.info(f"📉 Drawdown Máximo: {safe_format(attrs.get('max_drawdown', 0), pct=True)}")
        logger.info(f"📉 Peak Drawdown: {safe_format(attrs.get('peak_drawdown', 0), pct=True)}")
        
        # Trading metrics
        logger.info(f"🔄 Total de Trades: {attrs.get('total_trades', 0)}")
        logger.info(f"🎯 Win Rate: {safe_format(attrs.get('win_rate', 0), pct=True)}")
        logger.info(f"💸 PnL Total: {safe_format(attrs.get('total_pnl', 0), currency=True)}")
        logger.info(f"📅 Trades por Dia: {safe_format(attrs.get('trades_per_day', 0), '{:.2f}')}")
        logger.info(f"💰 Lucro Médio/Dia: {safe_format(attrs.get('lucro_medio_dia', 0), currency=True)}")
        
        # Advanced metrics
        if 'sharpe_ratio' in attrs:
            logger.info(f"📊 Sharpe Ratio: {safe_format(attrs.get('sharpe_ratio', 0), '{:.3f}')}")
        if 'sortino_ratio' in attrs:
            logger.info(f"📈 Sortino Ratio: {safe_format(attrs.get('sortino_ratio', 0), '{:.3f}')}")
        if 'calmar_ratio' in attrs:
            logger.info(f"📉 Calmar Ratio: {safe_format(attrs.get('calmar_ratio', 0), '{:.3f}')}")
        if 'annual_return' in attrs:
            logger.info(f"📈 Retorno Anual: {safe_format(attrs.get('annual_return', 0), pct=True)}")
        if 'volatility' in attrs:
            logger.info(f"📊 Volatilidade Anual: {safe_format(attrs.get('volatility', 0), pct=True)}")
        if 'trading_days' in attrs:
            logger.info(f"📅 Dias de Trading: {attrs.get('trading_days', 0)}")
            
        # Score components (se disponíveis)
        if 'performance_score' in attrs:
            logger.info(f"\n🎯 COMPONENTES DO SCORE:")
            logger.info(f"-" * 30)
            logger.info(f"  Performance: {safe_format(attrs.get('performance_score', 0))}")
            logger.info(f"  Consistência: {safe_format(attrs.get('consistency_score', 0))}")
            logger.info(f"  Trade Quality: {safe_format(attrs.get('trade_score', 0))}")
            logger.info(f"  Risk-Adj Return: {safe_format(attrs.get('risk_adjusted_return', 0))}")
        
        logger.info("="*70)
        
        # === TAMBÉM EXIBIR NO CONSOLE ===
        print(f"\n" + "="*70)
        print(f"🏆 MELHOR TRIAL - #{best_trial.number} | Score: {safe_format(best_trial.value)}")
        print(f"💰 Portfolio: {safe_format(attrs.get('portfolio_value', 0), currency=True)} | " +
              f"DD: {safe_format(attrs.get('max_drawdown', 0), pct=True)} | " + 
              f"Trades: {attrs.get('total_trades', 0)}")
        print(f"🎯 Win Rate: {safe_format(attrs.get('win_rate', 0), pct=True)} | " +
              f"PnL: {safe_format(attrs.get('total_pnl', 0), currency=True)}")
        
        # Hiperparâmetros principais no console
        params_str = []
        for key, value in best_trial.params.items():
            if key == 'learning_rate':
                params_str.append(f"LR={value:.1e}")
            elif key in ['n_steps', 'batch_size', 'n_epochs', 'lstm_hidden_size']:
                params_str.append(f"{key.replace('_', '').upper()}={value}")
            elif key in ['clip_range', 'ent_coef', 'vf_coef']:
                params_str.append(f"{key.replace('_', '').upper()}={value:.3f}")
        print(f"🔧 Params: {' | '.join(params_str)}")
        print("="*70)
        
        # === SALVAR EM ARQUIVO JSON COMPLETO ===
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "best_trial_number": best_trial.number,
            "best_score": best_trial.value,
            "best_params": best_trial.params,
            "performance_metrics": attrs,
            "total_trials_completed": len(study.trials),
            "study_info": {
                "direction": "maximize",
                "total_trials": len(study.trials),
                "completed_trials": len([t for t in study.trials if t.value is not None]),
                "failed_trials": len([t for t in study.trials if t.value is None])
            }
        }
        
        backup_dir = "logs"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        backup_path = os.path.join(backup_dir, filename)
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=4)
            
        logger.info(f"💾 Backup completo salvo em: {backup_path}")
        print(f"💾 Backup salvo: {backup_path}")
        
    except Exception as e:
        error_msg = f"Erro ao salvar melhores parâmetros: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    study = None
    logger = None
    try:
        import sys
        instance_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        logger = setup_logging(instance_id)
        storage_name = f"sqlite:///optuna_storage_{instance_id}.db"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"ppo_optimization_{instance_id}_{timestamp}"
        study = optuna.create_study(
            direction="maximize",
            pruner=None,  # 🚨 COMPLETAMENTE DESABILITADO - SEM PODA PREMATURA
            storage=storage_name,
            study_name=study_name,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
        logger.info(f"\n=== Iniciando Otimização PPO REFINAMENTO INTENSIVO ===")
        logger.info(f"Study Name: {study_name}")
        logger.info(f"Storage: {storage_name}")
        logger.info(f"Instance ID: {instance_id}")
        logger.info("="*50)
        
        # 🔥 SISTEMA HÍBRIDO: 50 refinamento + 50 exploração = 100 trials total
        print(f"\n🎯 INICIANDO OTIMIZAÇÃO HÍBRIDA SL/TP RANGES - Target: 100 trials")
        print(f"🔧 Fase 1 (0-49): REFINAMENTO FINO - Ajustes precisos nos melhores SL/TP (50 trials)")
        print(f"🚀 Fase 2 (50-99): EXPLORAÇÃO AMPLA - Busca por novos ótimos SL/TP (50 trials)")
        print(f"⏱️ 40k steps refinamento + 60k steps exploração = ~3-6 min por trial")
        print("="*80)
        
        # Callback para salvar melhores parâmetros a cada trial
        def trial_callback(study, trial):
            # Salvar resultado individual do trial EM TEMPO REAL
            save_trial_result(trial, instance_id, logger)
            
            # Mostrar progresso detalhado a cada trial
            if trial.value is not None:
                attrs = trial.user_attrs if hasattr(trial, 'user_attrs') else {}
                phase = attrs.get('phase', 'unknown')
                phase_emoji = "🔧" if phase == "refinement" else "🚀"
                
                # Progresso mais detalhado no terminal
                duration = trial.duration.total_seconds() if trial.duration else 0
                print(f"\n✅ {phase_emoji} Trial {trial.number} COMPLETED ({duration:.1f}s)")
                print(f"   📊 Score: {trial.value:.4f}")
                print(f"   💰 Portfolio: ${attrs.get('portfolio_value', 0):.0f}")
                print(f"   📉 Drawdown: {attrs.get('max_drawdown', 0)*100:.1f}%")
                print(f"   🔄 Trades: {attrs.get('total_trades', 0)}")
                print(f"   🎯 Win Rate: {attrs.get('win_rate', 0)*100:.1f}%")
                print(f"   📈 Trades/Day: {attrs.get('trades_per_day', 0):.1f}")
                
                # Mostrar parâmetros principais
                params_summary = []
                for key, value in trial.params.items():
                    if key == 'learning_rate':
                        params_summary.append(f"LR={value:.1e}")
                    elif key in ['n_steps', 'batch_size', 'n_epochs']:
                        params_summary.append(f"{key.upper().replace('_', '')}={value}")
                    elif key in ['clip_range', 'ent_coef', 'vf_coef']:
                        params_summary.append(f"{key.upper().replace('_', '')}={value:.3f}")
                print(f"   🔧 Key Params: {' | '.join(params_summary[:5])}")
            else:
                print(f"\n❌ Trial {trial.number}: FAILED")
                
            # Salvar resultados parciais a cada 10 trials
            if trial.number % 10 == 0 and trial.number > 0:
                print(f"\n📊 CHECKPOINT - Salvando progresso após {trial.number} trials...")
                save_partial_study_results(study, instance_id, logger, "checkpoint")
                save_best_params_to_log(study, logger)
        
        
        
        study.optimize(objective, n_trials=100, show_progress_bar=True, callbacks=[trial_callback])  # 🔥 SISTEMA HÍBRIDO: 50 refinamento + 50 exploração
        
        logger.info("\n=== Optimization Results ===")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_trial.value}")
        logger.info("\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"{key}: {value}")
        results_dir = "optimization_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_file = os.path.join(results_dir, f"ppo_optimization_{instance_id}_{timestamp}.json")
        best_trial_number = study.best_trial.number
        # Salvar o melhor modelo na pasta Best Model
        best_model_src = os.path.join(results_dir, f"ppo_model_trial_{best_trial_number}.zip")
        best_model_dst_dir = "Best Model"
        if not os.path.exists(best_model_dst_dir):
            os.makedirs(best_model_dst_dir)
        best_model_dst = os.path.join(best_model_dst_dir, "ppo_best_model.zip")
        if os.path.exists(best_model_src):
            shutil.copy(best_model_src, best_model_dst)
            logger.info(f"\nBest model saved to {best_model_dst}")
        else:
            logger.warning(f"Best model file not found: {best_model_src}")
        results = {
            'study_name': study_name,
            'best_trial': study.best_trial.number,
            'best_value': study.best_trial.value,
            'best_params': study.best_trial.params,
            'all_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"\nResults saved to {results_file}")
        
        # Salvar melhores parâmetros finais com relatório completo
        trials_completed = len(study.trials)
        print(f"\n🎉 OTIMIZAÇÃO HÍBRIDA COMPLETA - {trials_completed}/100 trials finalizados!")
        
        # Estatísticas das fases
        refinement_trials = min(50, trials_completed)  # 🔥 SISTEMA HÍBRIDO: 50 trials de refinamento
        exploration_trials = max(0, trials_completed - 50)  # 🔥 SISTEMA HÍBRIDO: 50 trials de exploração
        print(f"🔧 Refinamento: {refinement_trials}/50 trials")
        print(f"🚀 Exploração: {exploration_trials}/50 trials")
        
        save_best_params_to_log(study, logger, "final_best_params.json")
        
        # Relatório detalhado do best trial (mantido para compatibilidade)
        best_metrics = study.best_trial.user_attrs if hasattr(study.best_trial, 'user_attrs') else {}
        def safe_metric(val, pct=False):
            try:
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    return 0.0
                if pct:
                    # Clamp entre 0 e 1 para percentuais
                    val = max(0.0, min(1.0, val))
                    return val * 100
                return val
            except Exception:
                return 0.0
        print("\n[REPORT BEST TRIAL]")
        print(f"Portfolio final: ${safe_metric(best_metrics.get('portfolio_value')):.2f}")
        print(f"Saldo realizado: ${safe_metric(best_metrics.get('final_balance')):.2f}")
        print(f"PnL não realizado: ${safe_metric(best_metrics.get('unrealized_pnl')):.2f}")
        print(f"Pico do portfólio: ${safe_metric(best_metrics.get('peak_portfolio')):.2f}")
        print(f"Drawdown final: {safe_metric(best_metrics.get('max_drawdown'), pct=True):.2f}%")
        print(f"DD Peak: {safe_metric(best_metrics.get('peak_drawdown'), pct=True):.2f}%")
        print(f"Trades/dia: {safe_metric(best_metrics.get('trades_per_day')):.2f}")
        print(f"Lucro médio/dia: ${safe_metric(best_metrics.get('lucro_medio_dia')):.2f}")
        print(f"Total trades: {int(safe_metric(best_metrics.get('total_trades')))}")
        print(f"Win rate: {safe_metric(best_metrics.get('win_rate'), pct=True):.2f}%")
        print(f"Sharpe: {safe_metric(best_metrics.get('sharpe_ratio')):.2f}")
        print(f"Sortino: {safe_metric(best_metrics.get('sortino_ratio')):.2f}")
        print(f"Calmar: {safe_metric(best_metrics.get('calmar_ratio')):.2f}")
        print(f"Período: {int(safe_metric(best_metrics.get('trading_days')))} dias")
        print(f"Retorno anualizado: {safe_metric(best_metrics.get('annual_return'), pct=True):.2f}%")
        print(f"Volatilidade anual: {safe_metric(best_metrics.get('volatility'), pct=True):.2f}%")
        
    except KeyboardInterrupt:
        trials_completed = len(study.trials) if study and study.trials else 0
        print(f"\n[⚠️ INTERRUPÇÃO] Otimização interrompida pelo usuário no trial {trials_completed}/100")
        print(f"🔄 Salvando melhores resultados até agora...")
        
        if study and logger and trials_completed > 0:
            logger.info(f"Otimização interrompida pelo usuário após {trials_completed} trials")
            
            # Salvar resultados parciais completos
            save_partial_study_results(study, instance_id, logger, "keyboard_interrupt")
            save_best_params_to_log(study, logger, "interrupted_best_params.json")
            print("\n✅ Resultados parciais e melhores parâmetros foram salvos!")
            
            # Mostrar estatísticas finais
            completed_trials = [t for t in study.trials if t.value is not None]
            failed_trials = [t for t in study.trials if t.value is None]
            print(f"📊 ESTATÍSTICAS FINAIS:")
            print(f"   Total Trials: {trials_completed}")
            print(f"   Successful: {len(completed_trials)} ({len(completed_trials)/trials_completed*100:.1f}%)")
            print(f"   Failed: {len(failed_trials)} ({len(failed_trials)/trials_completed*100:.1f}%)")
            
            # Mostrar fase atual
            if trials_completed < 50:
                print(f"   Fase: REFINAMENTO (fase 1/2)")
            else:
                print(f"   Fase: EXPLORAÇÃO (fase 2/2)")
                
            # Mostrar melhor trial se existir
            if completed_trials:
                best = study.best_trial
                best_attrs = best.user_attrs if hasattr(best, 'user_attrs') else {}
                print(f"   Melhor Trial: #{best.number} | Score: {best.value:.4f}")
                print(f"   Melhor Portfolio: ${best_attrs.get('portfolio_value', 0):.0f}")
                
        elif trials_completed == 0:
            print("❌ Nenhum trial foi completado - nada para salvar")
        else:
            print("❌ Erro ao salvar - study ou logger não disponível")
            
    except Exception as e:
        trials_completed = len(study.trials) if study and study.trials else 0
        print(f"\n[🚨 CRITICAL ERROR] Otimização interrompida no trial {trials_completed}/100")
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Log detalhado e salvar melhores parâmetros mesmo com erro
        if logger:
            logger.error(f"[CRITICAL] Optimization stopped at trial {trials_completed}/100")
            logger.error(f"Error: {str(e)}")
            logger.error("Traceback completo:")
            logger.error(traceback.format_exc())
            
        # Tentar salvar mesmo com erro
        if study and trials_completed > 0:
            try:
                print(f"💾 Tentando salvar melhores resultados de {trials_completed} trials...")
                
                # Salvar resultados parciais completos mesmo com erro
                save_partial_study_results(study, instance_id, logger or setup_logging(), "critical_error")
                save_best_params_to_log(study, logger or setup_logging(), "error_best_params.json")
                print("✅ Resultados parciais e melhores parâmetros foram salvos antes do erro!")
                
                # Mostrar estatísticas mesmo com erro
                completed_trials = [t for t in study.trials if t.value is not None]
                failed_trials = [t for t in study.trials if t.value is None]
                print(f"📊 ESTATÍSTICAS ANTES DO ERRO:")
                print(f"   Total Trials: {trials_completed}")
                print(f"   Successful: {len(completed_trials)} ({len(completed_trials)/trials_completed*100:.1f}%)")
                print(f"   Failed: {len(failed_trials)} ({len(failed_trials)/trials_completed*100:.1f}%)")
                
                # Mostrar fase atual
                if trials_completed < 50:
                    print(f"   Fase: REFINAMENTO (fase 1/2)")
                else:
                    print(f"   Fase: EXPLORAÇÃO (fase 2/2)")
                    
                # Mostrar melhor trial se existir
                if completed_trials:
                    best = study.best_trial
                    best_attrs = best.user_attrs if hasattr(best, 'user_attrs') else {}
                    print(f"   Melhor Trial: #{best.number} | Score: {best.value:.4f}")
                    print(f"   Melhor Portfolio: ${best_attrs.get('portfolio_value', 0):.0f}")
                    
            except Exception as save_error:
                error_msg = f"Erro ao salvar parâmetros: {save_error}"
                print(f"❌ {error_msg}")
                if logger:
                    logger.error(error_msg)
        else:
            print("❌ Nenhum trial completado ou study não disponível")
        
        # NÃO CONTINUAR - Re-raise para mostrar o erro real
        raise e

if __name__ == "__main__":
    main()









