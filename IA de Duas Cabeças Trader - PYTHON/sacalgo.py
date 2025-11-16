"""
🚀 MASSIVE SAC TRAINER - VERSÃO ULTRA SIMPLIFICADA
APENAS TREINAMENTO DIRETO SEM FASES COMPLEXAS
"""

import os
import sys
import time
import logging
import warnings
import traceback
import gc
import json
import pandas as pd
import numpy as np
import gym
from gym import spaces
from sklearn.impute import KNNImputer
import torch
import torch.nn as nn
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.cuda.amp import autocast, GradScaler  # 🔥 AMP IMPORTS

# Importar sistema de recompensa 
try:
    from reward_system import create_reward_system, CLASSIC_CONFIG
except ImportError:
    print("[WARNING] Não foi possível importar reward_system. Usando recompensa básica.")
    def create_reward_system(*args, **kwargs):
        return None
    CLASSIC_CONFIG = {}

# === 🔥 PARÂMETROS OTIMIZADOS FIXOS PARA SAC ===
USE_VECNORM = True

# 🔥 HIPERPARÂMETROS OTIMIZADOS PARA SAC
BEST_PARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 1000000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "policy_kwargs": {
        "net_arch": [256, 256],
        "activation_fn": torch.nn.ReLU,
        "use_sde": False,
    }
}

# 🔥 CONFIGURAÇÃO AMP (AUTOMATIC MIXED PRECISION)
ENABLE_AMP = torch.cuda.is_available()  # Só ativar se GPU disponível
if ENABLE_AMP:
    print("AMP (Automatic Mixed Precision) ATIVADO no SAC TRAINER - Treinamento acelerado!")
    # Configurar PyTorch para usar AMP
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 para matmul
else:
    print("⚠️ AMP desabilitado no SAC TRAINER - GPU não disponível")

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch.nn.functional as F
    FRAMEWORK_AVAILABLE = True
except ImportError:
    print("⚠️ Algumas dependências não estão disponíveis. Funcionalidade limitada.")
    FRAMEWORK_AVAILABLE = False

# === CLASSES NECESSÁRIAS ===

# 🔥 IMPORTAR SACTWOHEADPOLICY CUSTOMIZADA
from sactwohead import SACTwoHeadPolicy

def setup_gpu_optimized_sac():
    """Configurar GPU com otimizações para AMP no SAC TRAINER"""
    if torch.cuda.is_available():
        # 🔥 AMP: Configurações otimizadas para mixed precision
        torch.backends.cudnn.benchmark = True  # Otimizar para tamanhos fixos
        torch.backends.cudnn.allow_tf32 = True  # Permitir TF32 para Ampere
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 para matmul
        
        # Configurar memória GPU
        torch.cuda.empty_cache()
        
        # Log das configurações
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🚀 SAC TRAINER - GPU OTIMIZADA PARA AMP:")
        print(f"   Device: {device_name}")
        print(f"   Memória: {memory_total:.1f} GB")
        print(f"   CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        
        return True
    else:
        print("⚠️ SAC TRAINER - GPU não disponível - usando CPU")
        return False

class TradingEnv(gym.Env):
    MAX_STEPS = 200000  # 🔥 AUMENTADO: Episódios de 200k steps para aprendizado efetivo
    
    def __init__(self, df, window_size=20, is_training=True, initial_balance=1000):
        super(TradingEnv, self).__init__()
        
        # 🚨 VERIFICAÇÃO DE SEGURANÇA: Dataset não pode estar vazio
        if df is None or len(df) == 0:
            raise ValueError("❌ Dataset está vazio ou None! Verifique o arquivo de dados.")
        
        # 🔥 USAR 100% DOS DADOS PARA TREINAMENTO - dataset de validação já é separado
        self.df = df.copy()  # Usar dataset completo
        print(f"[ENV] Inicializando com {len(self.df):,} registros")
        self.window_size = window_size
        self.current_step = window_size
        self.initial_balance = initial_balance
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.positions = []
        self.returns = []
        self.trades = []  # Garantir que seja uma lista
        self.start_date = pd.to_datetime(self.df.index[0])
        self.end_date = pd.to_datetime(self.df.index[-1])
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        self.max_lot_size = 0.08  # 🔥 ALINHADO: Max lot size conforme solicitado
        self.max_positions = 3    # 🔥 ALINHADO: 3 posições máximas conforme solicitado
        self.current_positions = 0
        
        # 🔥 ACTION SPACE ATUALIZADO PARA SAC: Ações contínuas
        # estratégica: [-1,1] onde -1=short, 0=hold, 1=long
        # tática para cada posição: [-1,1] onde -1=close, 0=hold, 1=adjust
        # sltp para cada posição: valores contínuos [-3,3] para SL/TP significativos
        self.action_space = spaces.Box(
            low=np.array([-1] + [-1]*3 + [-3]*6, dtype=np.float32),   # estratégica + 3 táticas + 6 sltp (3x2)
            high=np.array([1] + [1]*3 + [3]*6, dtype=np.float32),     # estratégica + 3 táticas + 6 sltp (3x2)
            dtype=np.float32
        )
        
        self.imputer = KNNImputer(n_neighbors=5)
        base_features = [
            'returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 'stoch_k', 'volume_ratio', 'var_99', 'atr_14'
        ]
        self.feature_columns = []
        for tf in ['5m', '15m', '4h']:
            self.feature_columns.extend([f"{f}_{tf}" for f in base_features])
        self._prepare_data()
        # 🔥 OBSERVATION SPACE ATUALIZADO: 3 posições conforme solicitado
        n_features = len(self.feature_columns) + self.max_positions * 7  # 3 posições * 7 features cada
        self.observation_space = spaces.Box(
            low=np.full((window_size * n_features,), -np.inf, dtype=np.float32), 
            high=np.full((window_size * n_features,), np.inf, dtype=np.float32), 
            dtype=np.float32
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
        self.lot_size = 0.05  # 🔥 ALINHADO: Lot size conforme solicitado
        self.steps_since_last_trade = 0
        self.INACTIVITY_THRESHOLD = 24  # ~2h em 5m
        self.last_action = None
        self.hold_count = 0
        
        # 🚀 SUPER MODELO: Usar sistema de recompensas otimizado
        self.reward_system = create_reward_system("classic", initial_balance, CLASSIC_CONFIG)
        
        # 🔥 CONFIGURAÇÕES FINAIS ALINHADAS
        self.max_lot_size = 0.08  # 🔥 ALINHADO: Max lot size conforme solicitado  
        self.max_positions = 3    # 🔥 ALINHADO: 3 posições máximas conforme solicitado

    def reset(self, **kwargs):
        """Reset do ambiente para um novo episódio."""
        # 🔥 DATASET LOOP: Permitir reset em qualquer ponto do dataset
        if len(self.df) > self.window_size * 2:
            # Escolher ponto aleatório, mas deixar espaço para window_size e pelo menos 1000 steps
            max_start = len(self.df) - max(1000, self.window_size + 100)
            if max_start > self.window_size:
                self.current_step = np.random.randint(self.window_size, max_start)
            else:
                self.current_step = self.window_size
        else:
            self.current_step = self.window_size
        
        # Reset das variáveis de estado
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.positions = []
        self.returns = []
        self.trades = []
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        self.current_positions = 0
        self.win_streak = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.realized_balance = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.last_trade_pnl = 0.0
        self.steps_since_last_trade = 0
        self.last_action = None
        self.hold_count = 0
        
        return self._get_observation()

    def step(self, action):
        """Executar uma ação no ambiente."""
        # 🔥 SAC: Converter ações contínuas para discretas
        strategic_action = action[0]
        tactical_actions = action[1:4]
        sltp_actions = action[4:10].reshape(3, 2)  # 3 posições, 2 valores cada (SL, TP)
        
        # Converter ação estratégica contínua para discreta
        if strategic_action < -0.33:
            strategic_discrete = 2  # short
        elif strategic_action > 0.33:
            strategic_discrete = 1  # long
        else:
            strategic_discrete = 0  # hold
        
        # Converter ações táticas contínuas para discretas
        tactical_discrete = []
        for tact in tactical_actions:
            if tact < -0.33:
                tactical_discrete.append(1)  # close
            elif tact > 0.33:
                tactical_discrete.append(2)  # adjust
            else:
                tactical_discrete.append(0)  # hold
        
        # Reconstruir ação no formato esperado
        discrete_action = np.array([strategic_discrete] + tactical_discrete + list(sltp_actions.flatten()), dtype=np.float32)
        
        old_state = {
            "portfolio_total_value": self.realized_balance + sum(self._get_position_pnl(pos, self.df.iloc[self.current_step]['close_5m']) for pos in self.positions),
            "current_drawdown": self.current_drawdown
        }
        
        self.episode_steps += 1
        
        # 🔥 DATASET LOOP: Verificar se chegou ao fim e fazer loop
        if self.current_step >= len(self.df) - 1:
            self.current_step = self.window_size  # Voltar ao início
            print(f"[ENV] Dataset loop - voltando ao início (step {self.current_step})")
        
        current_price = self.df.iloc[self.current_step]['close_5m']
        
        # Processar ação estratégica
        if strategic_discrete == 1 and len(self.positions) < self.max_positions:  # Long
            position = {
                'type': 'long',
                'entry_price': current_price,
                'entry_step': self.current_step,
                'lot_size': self.lot_size,
                'sl': current_price * (1 - 0.02),  # 2% SL inicial
                'tp': current_price * (1 + 0.04),  # 4% TP inicial
                'exit_step': None,
                'exit_price': None,
                'pnl': 0
            }
            self.positions.append(position)
            self.current_positions += 1
            
        elif strategic_discrete == 2 and len(self.positions) < self.max_positions:  # Short
            position = {
                'type': 'short',
                'entry_price': current_price,
                'entry_step': self.current_step,
                'lot_size': self.lot_size,
                'sl': current_price * (1 + 0.02),  # 2% SL inicial
                'tp': current_price * (1 - 0.04),  # 4% TP inicial
                'exit_step': None,
                'exit_price': None,
                'pnl': 0
            }
            self.positions.append(position)
            self.current_positions += 1
        
        # Processar ações táticas e SL/TP para posições existentes
        positions_to_remove = []
        for i, pos in enumerate(self.positions):
            if i < len(tactical_discrete):
                # Ação tática
                if tactical_discrete[i] == 1:  # Close position
                    pnl = self._get_position_pnl(pos, current_price)
                    pos['exit_step'] = self.current_step
                    pos['exit_price'] = current_price
                    pos['pnl'] = pnl
                    self.portfolio_value += pnl
                    self.trades.append(pos.copy())
                    positions_to_remove.append(i)
                    self.current_positions -= 1
                    self.last_trade_pnl = pnl
                    
                elif tactical_discrete[i] == 2 and i < len(sltp_actions):  # Adjust SL/TP
                    sl_adj, tp_adj = sltp_actions[i]
                    # 🔥 CORREÇÃO: Garantir que sl e tp existem antes de modificar
                    if 'sl' not in pos:
                        pos['sl'] = pos['entry_price'] * 0.98 if pos['type'] == 'long' else pos['entry_price'] * 1.02
                    if 'tp' not in pos:
                        pos['tp'] = pos['entry_price'] * 1.04 if pos['type'] == 'long' else pos['entry_price'] * 0.96
                    
                    if pos['type'] == 'long':
                        pos['sl'] = current_price * (1 + sl_adj * 0.01)  # Ajuste em %
                        pos['tp'] = current_price * (1 + tp_adj * 0.01)
                    else:  # short
                        pos['sl'] = current_price * (1 - sl_adj * 0.01)
                        pos['tp'] = current_price * (1 - tp_adj * 0.01)
        
        # Remover posições fechadas
        for i in reversed(positions_to_remove):
            self.positions.pop(i)
        
        # Verificar SL/TP automático
        positions_to_remove = []
        for i, pos in enumerate(self.positions):
            hit_sl = False
            hit_tp = False
            
            if pos['type'] == 'long':
                hit_sl = current_price <= pos.get('sl', 0) if pos.get('sl', 0) > 0 else False
                hit_tp = current_price >= pos.get('tp', 0) if pos.get('tp', 0) > 0 else False
            else:  # short
                hit_sl = current_price >= pos.get('sl', 0) if pos.get('sl', 0) > 0 else False
                hit_tp = current_price <= pos.get('tp', 0) if pos.get('tp', 0) > 0 else False
            
            if hit_sl or hit_tp:
                pnl = self._get_position_pnl(pos, current_price)
                pos['exit_step'] = self.current_step
                pos['exit_price'] = current_price
                pos['pnl'] = pnl
                self.portfolio_value += pnl
                self.trades.append(pos.copy())
                positions_to_remove.append(i)
                self.current_positions -= 1
                self.last_trade_pnl = pnl
        
        # Remover posições que atingiram SL/TP
        for i in reversed(positions_to_remove):
            self.positions.pop(i)
        
        # Atualizar portfolio com PnL não realizado
        unrealized_pnl = sum(self._get_position_pnl(pos, current_price) for pos in self.positions)
        current_portfolio = self.portfolio_value + unrealized_pnl
        
        # Atualizar métricas
        if current_portfolio > self.peak_portfolio:
            self.peak_portfolio = current_portfolio
        
        self.current_drawdown = (self.peak_portfolio - current_portfolio) / self.peak_portfolio
        if self.current_drawdown > self.peak_drawdown:
            self.peak_drawdown = self.current_drawdown
        
        # Calcular reward
        reward, info = self._calculate_reward_and_info(discrete_action, old_state)
        
        # Verificar se episódio terminou
        done = (self.episode_steps >= self.MAX_STEPS or 
                current_portfolio <= self.initial_balance * 0.5 or  # Stop loss de 50%
                self.current_step >= len(self.df) - 1)
        
        self.current_step += 1
        
        return self._get_observation(), reward, done, info

    def _prepare_data(self):
        # Renomear colunas close duplicadas para cada timeframe, se necessário
        if 'close' in self.df.columns:
            close_cols = [col for col in self.df.columns if col == 'close']
            if len(close_cols) > 1:
                # Se houver múltiplas colunas 'close', renomear para close_5m, close_15m, close_4h
                close_names = ['close_5m', 'close_15m', 'close_4h']
                for i, col in enumerate(close_cols):
                    self.df.rename(columns={col: close_names[i]}, inplace=True, level=0)
        # Se só existe uma coluna 'close', renomear para 'close_5m'
        if 'close' in self.df.columns and 'close_5m' not in self.df.columns:
            self.df.rename(columns={'close': 'close_5m'}, inplace=True)
        # Calcular features técnicas básicas para cada timeframe
        for tf in ['5m', '15m', '4h']:
            close_col = f'close_{tf}'
            if close_col in self.df.columns:
                # returns
                self.df.loc[:, f'returns_{tf}'] = self.df[close_col].pct_change().fillna(0)
                # volatility_20
                self.df.loc[:, f'volatility_20_{tf}'] = self.df[close_col].rolling(window=20).std().fillna(0)
                # sma_20
                self.df.loc[:, f'sma_20_{tf}'] = self.df[close_col].rolling(window=20).mean().bfill().fillna(0)
                # sma_50
                self.df.loc[:, f'sma_50_{tf}'] = self.df[close_col].rolling(window=50).mean().bfill().fillna(0)
                # rsi_14
                try:
                    import ta
                    self.df.loc[:, f'rsi_14_{tf}'] = ta.momentum.RSIIndicator(self.df[close_col], window=14).rsi().fillna(0)
                except Exception:
                    self.df.loc[:, f'rsi_14_{tf}'] = 0
                # stoch_k
                try:
                    self.df.loc[:, f'stoch_k_{tf}'] = ta.momentum.StochasticOscillator(self.df[close_col], self.df[close_col], self.df[close_col], window=14).stoch().fillna(0)
                except Exception:
                    self.df.loc[:, f'stoch_k_{tf}'] = 0
                # volume_ratio (não há volume, então zero)
                self.df.loc[:, f'volume_ratio_{tf}'] = 0
                # var_99
                self.df.loc[:, f'var_99_{tf}'] = self.df[close_col].rolling(window=20).quantile(0.01).fillna(0)
                # atr_14
                try:
                    self.df.loc[:, f'atr_14_{tf}'] = ta.volatility.AverageTrueRange(self.df[close_col], self.df[close_col], self.df[close_col], window=14).average_true_range().fillna(0)
                except Exception:
                    self.df.loc[:, f'atr_14_{tf}'] = 0
                # sma_cross
                self.df.loc[:, f'sma_cross_{tf}'] = (self.df[f'sma_20_{tf}'] > self.df[f'sma_50_{tf}']).astype(float) - (self.df[f'sma_20_{tf}'] < self.df[f'sma_50_{tf}']).astype(float)
                # momentum_5
                self.df.loc[:, f'momentum_5_{tf}'] = self.df[close_col].pct_change(periods=5).fillna(0)
        # Criar colunas ausentes como zero
        for col in self.feature_columns:
            if col not in self.df.columns:
                self.df.loc[:, col] = 0
        for col in self.feature_columns:
            self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            self.df.loc[:, col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        base_imputer = KNNImputer(n_neighbors=5)
        base_imputed = base_imputer.fit_transform(self.df[self.feature_columns])
        if base_imputed.shape[1] != len(self.feature_columns):
            raise ValueError('Shape mismatch entre base_imputed e feature_columns')
        self.df.loc[:, self.feature_columns] = pd.DataFrame(base_imputed, index=self.df.index, columns=self.feature_columns)
        self.processed_data = self.df[self.feature_columns].values
        if np.any(np.isnan(self.processed_data)) or np.any(np.isinf(self.processed_data)):
            self.processed_data = np.nan_to_num(self.processed_data, nan=0.0, posinf=1e6, neginf=-1e6)
        # Feature binária de oportunidade (apenas para 5m)
        self.df['opportunity'] = 0
        if 'sma_cross_5m' in self.df.columns:
            cross = self.df['sma_cross_5m']
            self.df['opportunity'] = ((cross.shift(1) != cross) & (cross != 0)).astype(int)

    def _get_observation(self):
        # 🔥 DATASET LOOP: Corrigir current_step se necessário
        if self.current_step < self.window_size:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        positions_obs = np.zeros((self.max_positions, 7))
        current_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
        
        for i in range(self.max_positions):
            if i < len(self.positions):
                pos = self.positions[i]
                positions_obs[i, 0] = 1  # status aberta
                positions_obs[i, 1] = 0 if pos['type'] == 'long' else 1
                positions_obs[i, 2] = (pos['entry_price'] - min(self.df[f'close_{self.base_tf}'])) / (max(self.df[f'close_{self.base_tf}']) - min(self.df[f'close_{self.base_tf}']))
                # PnL atual
                if pos['type'] == 'long':
                    pnl = (current_price - pos['entry_price']) * pos['lot_size']
                else:
                    pnl = (pos['entry_price'] - current_price) * pos['lot_size']
                positions_obs[i, 3] = pnl
                positions_obs[i, 4] = pos.get('sl', 0)
                positions_obs[i, 5] = pos.get('tp', 0)
                positions_obs[i, 6] = (self.current_step - pos['entry_step']) / len(self.df)
            else:
                positions_obs[i, :] = 0  # slot vazio
                
        obs_market = self.processed_data[self.current_step - self.window_size:self.current_step]
        tile_positions = np.tile(positions_obs.flatten(), (self.window_size, 1))
        
        assert obs_market.shape[0] == tile_positions.shape[0], f"obs_market shape: {obs_market.shape}, tile_positions shape: {tile_positions.shape}"
        obs = np.concatenate([obs_market, tile_positions], axis=1)
        flat_obs = obs.flatten().astype(np.float32)
        
        assert isinstance(flat_obs, np.ndarray), f"flat_obs não é np.ndarray: {type(flat_obs)}"
        assert flat_obs.ndim == 1, f"flat_obs não é 1D: shape={flat_obs.shape}"
        assert flat_obs.shape == self.observation_space.shape, f"flat_obs.shape {flat_obs.shape} != observation_space.shape {self.observation_space.shape}"
        assert flat_obs.dtype == np.float32, f"flat_obs.dtype {flat_obs.dtype} != np.float32"
        assert all(isinstance(x, (float, np.floating, np.float32, np.float64)) for x in flat_obs[:10]), f"flat_obs contém elementos não-float nos primeiros 10: {[type(x) for x in flat_obs[:10]]}"
        
        return flat_obs

    def _calculate_reward_and_info(self, action, old_state):
        # Usar sistema de recompensa modular se disponível
        if self.reward_system:
            reward, info, _ = self.reward_system.calculate_reward_and_info(self, action, old_state)
            
            # 🔥 CORREÇÃO CRÍTICA: Garantir que sempre há alguma recompensa para gradientes
            if abs(reward) < 1e-8:
                # Recompensa mínima baseada na variação do preço
                current_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
                if self.current_step > self.window_size:
                    prev_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step - 1]
                    price_change = (current_price - prev_price) / prev_price
                    reward = price_change * 0.01  # Pequena recompensa baseada na mudança de preço
                else:
                    reward = -0.001  # Pequena penalidade padrão
            
            return reward, info
        else:
            # Sistema de recompensa básico como fallback
            # 🔥 NUNCA retornar zero - sempre alguma variação
            current_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
            if self.current_step > self.window_size:
                prev_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step - 1]
                price_change = (current_price - prev_price) / prev_price
                reward = price_change * 0.01
            else:
                reward = -0.001
            
            return reward, {}

    def _get_position_pnl(self, pos, current_price):
        if pos['type'] == 'long':
            return (current_price - pos['entry_price']) * pos['lot_size']
        else:
            return (pos['entry_price'] - current_price) * pos['lot_size']

def make_wrapped_env(df, window_size, is_training, initial_portfolio=1000):
    """Criar ambiente com configurações otimizadas"""
    env = TradingEnv(
        df=df,
        window_size=window_size,
        is_training=is_training,
        initial_balance=initial_portfolio
    )
    
    # Configurar Box com float32 para evitar warning
    env.observation_space = gym.spaces.Box(
        low=np.full(env.observation_space.shape, -np.inf, dtype=np.float32), 
        high=np.full(env.observation_space.shape, np.inf, dtype=np.float32), 
        dtype=np.float32
    )
    
    env = Monitor(env)
    return env

def get_latest_processed_file(timeframe):
    # 🚀 BUSCAR DATASETS DISPONÍVEIS (prioridade: novos datasets > data/fixed)
    possible_paths = [
        f'novos datasets/GOLD_{timeframe}_processed.csv',
        f'novos datasets/GOLD_train.csv' if timeframe == 'train' else f'novos datasets/GOLD_val.csv',  # Dataset específico
        f'data/fixed/train.csv' if timeframe == 'train' else f'data/fixed/val.csv',  # Fallback
    ]
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            print(f"[INFO] Carregando dataset: {file_path}")
            try:
                # Tentar diferentes formatos de data
                if 'Date' in pd.read_csv(file_path, nrows=1).columns:
                    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                else:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                if len(df) > 0:
                    print(f"✅ Dataset carregado: {len(df):,} registros")
                    return df
                else:
                    print(f"[WARNING] Dataset {file_path} está vazio!")
            except Exception as e:
                print(f"[WARNING] Erro ao carregar {file_path}: {e}")
                continue
    
    print("❌ NENHUM DATASET VÁLIDO ENCONTRADO!")
    raise FileNotFoundError("Nenhum dataset encontrado!")

def find_existing_models():
    """🔍 Sistema de detecção de modelos SAC existentes para RESUME TRAINING"""
    print("\n🔍 PROCURANDO MODELOS SAC EXISTENTES PARA RESUME...")
    
    # 🏗️ SAC TRAINER: BUSCAR EM MÚLTIPLOS LOCAIS
    search_paths = [
        # Training SAC (novo diretório)
        "training SAC/checkpoints/checkpoint_*_steps_*.zip",
        "training SAC/checkpoints/EMERGENCY_*.zip",
        "training SAC/sac_checkpoint_*_steps.zip",
        "training SAC/sac_final_model_*.zip",
        "training SAC/sac_model_*.zip",
        "training SAC/trained_model_final.zip",
        "training SAC/emergency_model.zip",
        "training SAC/RESCUE_MODEL.zip",
        # Diretório raiz (checkpoints automáticos)
        "auto_checkpoint_*_steps.zip",
        # Best Model folder
        "Best Model/*.zip"
    ]
    
    print(f"🔍 Buscando em {len(search_paths)} padrões de arquivos...")
    for pattern in search_paths:
        print(f"  - {pattern}")
    
    existing_models = []
    
    # Buscar por padrões glob
    import glob
    for pattern in search_paths:
        matches = glob.glob(pattern)
        print(f"  📁 {pattern} -> {len(matches)} arquivos encontrados")
        for path in matches:
            if os.path.exists(path):
                try:
                    size = os.path.getsize(path)
                    size_mb = size / (1024 * 1024)
                    mtime = os.path.getmtime(path)
                    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                    
                    # Extrair steps do nome - múltiplos padrões
                    steps_trained = 0
                    filename = os.path.basename(path)
                    
                    # Padrão: auto_checkpoint_280000_steps.zip
                    if "auto_checkpoint_" in filename and "_steps.zip" in filename:
                        try:
                            steps_str = filename.split("auto_checkpoint_")[1].split("_steps.zip")[0]
                            steps_trained = int(steps_str)
                        except:
                            steps_trained = 0
                    
                    # Padrão: checkpoint_280000_steps_20250613_075912.zip
                    elif "checkpoint_" in filename and "_steps_" in filename:
                        try:
                            # Extrair número entre checkpoint_ e _steps_
                            parts = filename.split("checkpoint_")[1].split("_steps_")
                            steps_str = parts[0]
                            steps_trained = int(steps_str)
                        except:
                            steps_trained = 0
                    
                    existing_models.append({
                        'path': os.path.abspath(path),
                        'size_mb': size_mb,
                        'modified': mtime_str,
                        'mtime': mtime,
                        'steps_trained': steps_trained
                    })
                except Exception as e:
                    print(f"[WARNING] Erro ao analisar {path}: {e}")
    
    # Ordenar por steps treinados (maior primeiro), depois por data de modificação
    existing_models.sort(key=lambda x: (x['steps_trained'], x['mtime']), reverse=True)
    
    if existing_models:
        print(f"✅ ENCONTRADOS {len(existing_models)} MODELOS SAC:")
        for i, model in enumerate(existing_models, 1):
            print(f"  {i}. {os.path.basename(model['path'])}")
            print(f"     📁 {model['size_mb']:.1f} MB | 📅 {model['modified']}")
            if model['steps_trained'] > 0:
                print(f"     🚀 Steps treinados: {model['steps_trained']:,}")
        print()
        
        return existing_models
    else:
        print("❌ NENHUM MODELO SAC ENCONTRADO - Iniciando do zero")
        return []

def main():
    """
    🚀 MASSIVE SAC TRAINER - VERSÃO ULTRA SIMPLIFICADA COM RESUME
    Treinamento direto de 2,000,000 steps sem fases complexas
    """
    # Desabilitar warnings do TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # 🔥 AMP: Configurar GPU com otimizações
    gpu_available = setup_gpu_optimized_sac()
    if gpu_available and ENABLE_AMP:
        print("SAC TRAINER - AMP habilitado - Treinamento acelerado!")
    
    print("\n" + "="*60)
    print(" MASSIVE SAC TRAINER - VERSÃO ULTRA SIMPLIFICADA")
    print("="*60)
    print()
    print("[INFO] Iniciando treinamento massivo...")
    print("[CONFIG] 2M steps diretos, 3 posições max, lot 0.05")
    print("[DATASET] 1M+ barras treino + 400k+ eval")
    if ENABLE_AMP and gpu_available:
        print("[AMP] Mixed Precision ativado - 1.5-2x mais rápido!")
    print()
    print("Pressione Ctrl+C para parar o treinamento")
    print("="*60)
    print()
        
    try:
        # 🏗️ CONFIGURAÇÃO FRAMEWORK SAC
        os.makedirs("training SAC/models", exist_ok=True)
        os.makedirs("training SAC/checkpoints", exist_ok=True)
        os.makedirs("training SAC/logs", exist_ok=True)
        
        # 🔍 SISTEMA DE RESUME TRAINING
        existing_models = find_existing_models()
        model_to_load = None
        steps_already_trained = 0
        
        if existing_models:
            print("💡 OPÇÕES DE RESUME TRAINING:")
            print("0. ❌ Iniciar treinamento do ZERO (ignorar modelos)")
            
            for i, model in enumerate(existing_models[:5], 1):  # Mostrar top 5
                steps_info = f" ({model['steps_trained']:,} steps)" if model['steps_trained'] > 0 else ""
                print(f"{i}. ✅ Carregar {os.path.basename(model['path'])}{steps_info}")
            
            while True:
                try:
                    choice = input("\n🔥 Escolha uma opção (0-{max_choice}): ".format(max_choice=min(5, len(existing_models))))
                    choice = int(choice)
                    if choice == 0:
                        print("🚀 Iniciando do ZERO...")
                        break
                    elif 1 <= choice <= min(5, len(existing_models)):
                        model_to_load = existing_models[choice - 1]
                        steps_already_trained = model_to_load['steps_trained']
                        print(f"🔄 RESUME do modelo: {os.path.basename(model_to_load['path'])}")
                        if steps_already_trained > 0:
                            remaining_steps = max(0, 2000000 - steps_already_trained)
                            print(f"📊 Steps já treinados: {steps_already_trained:,}")
                            print(f"🎯 Steps restantes: {remaining_steps:,}")
                        break
                except ValueError:
                    print("❌ Opção inválida!")
        
        # Setup básico de logging no framework SAC
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training SAC/logs/sac_training.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # 🔥 TREINAMENTO DIRETO - PARÂMETROS FIXOS OTIMIZADOS
        # Carregar datasets de treino e validação
        df_train = get_latest_processed_file('train')
        df_val = get_latest_processed_file('val')
        
        # Criar ambiente de treino
        env_train = DummyVecEnv([lambda: make_wrapped_env(
            df=df_train, 
            window_size=20, 
            is_training=True,
            initial_portfolio=1000
        )])
        
        # Criar ambiente de validação
        env_val = DummyVecEnv([lambda: make_wrapped_env(
            df=df_val,
            window_size=20,
            is_training=False,
            initial_portfolio=1000
        )])
        
        if USE_VECNORM:
            vec_normalize_file = 'vec_normalize.pkl'
            if os.path.exists(vec_normalize_file):
                print(f"[INFO] Carregando VecNormalize de {vec_normalize_file}")
                env_train = VecNormalize.load(vec_normalize_file, env_train)
                env_train.training = True
                env_train.norm_reward = True
            else:
                print("[INFO] Criando novo VecNormalize")
                env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, gamma=0.99)
        
        # 🔄 CRIAR OU CARREGAR MODELO
        if model_to_load:
            print(f"🔄 Carregando modelo existente: {model_to_load['path']}")
            try:
                # 🔥 AMP: Configurar device para mixed precision
                device_policy = "cuda" if torch.cuda.is_available() else "cpu"
                model = SAC.load(model_to_load['path'], env=env_train, device=device_policy)
                model.set_env(env_train)  # Garantir que o ambiente esteja correto
                
                # 🔥 CORREÇÃO CRÍTICA: Garantir que modelo está em modo de treinamento
                model.policy.train()
                if hasattr(model, 'policy_old'):
                    model.policy_old.train()
                print("✅ Modelo configurado para modo TRAIN")
                
                # 🔥 CORREÇÃO CRÍTICA: Verificar e corrigir requires_grad
                params_fixed = 0
                for param in model.policy.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        params_fixed += 1
                if params_fixed > 0:
                    print(f"🔧 CORRIGIDO: {params_fixed} parâmetros tinham requires_grad=False")
                
                # 🔥 CORREÇÃO CRÍTICA: Recriar otimizador para garantir funcionamento
                if hasattr(model.policy, 'optimizer'):
                    old_lr = model.policy.optimizer.param_groups[0]['lr']
                    print(f"🔧 Recriando otimizador (LR anterior: {old_lr:.2e})")
                    
                    # Recriar otimizador com mesmos parâmetros
                    model.policy.optimizer = torch.optim.AdamW(
                        model.policy.parameters(),
                        lr=old_lr,
                        eps=1e-5,
                        weight_decay=0.01,
                        amsgrad=False
                    )
                    print("✅ Otimizador recriado com sucesso")
                
                # 🔥 AMP: Configurar GradScaler se AMP estiver habilitado
                if ENABLE_AMP and hasattr(model, 'policy'):
                    model._amp_scaler = GradScaler()
                    print("✅ GradScaler configurado para modelo carregado")
                
                print("✅ Modelo carregado com sucesso!")
                if ENABLE_AMP:
                    print("🚀 AMP configurado para modelo carregado!")
                
                # 🔥 CORREÇÃO CRÍTICA: Preservar num_timesteps do modelo carregado
                if hasattr(model, 'num_timesteps') and model.num_timesteps > 0:
                    steps_already_trained = model.num_timesteps
                    print(f"🔄 PROGRESSO PRESERVADO: {steps_already_trained:,} steps já treinados")
                elif steps_already_trained > 0:
                    # Se o modelo não tem num_timesteps mas sabemos quantos steps foram treinados
                    model.num_timesteps = steps_already_trained
                    print(f"🔧 PROGRESSO RESTAURADO: {steps_already_trained:,} steps definidos no modelo")
                
                # 🔥 VERIFICAÇÃO CRÍTICA: Verificar se modelo tem pesos válidos
                if hasattr(model.policy, 'state_dict'):
                    state_dict = model.policy.state_dict()
                    total_params = sum(p.numel() for p in state_dict.values())
                    non_zero_params = sum(torch.count_nonzero(p).item() for p in state_dict.values())
                    zero_ratio = (total_params - non_zero_params) / total_params
                    print(f"🔍 MODELO CARREGADO: {total_params:,} parâmetros, {non_zero_params:,} não-zero ({(1-zero_ratio)*100:.1f}%)")
                    
                    if zero_ratio > 0.9:
                        print("🚨 AVISO: Modelo carregado tem muitos parâmetros zero - pode estar corrompido!")
                
                # Atualizar tensorboard log
                model.tensorboard_log = 'training SAC/logs/tensorboard/'
            except Exception as e:
                print(f"❌ ERRO ao carregar modelo: {e}")
                print("🔄 Criando novo modelo...")
                
                # 🔥 AMP: Configurações otimizadas para novo modelo
                model_config = BEST_PARAMS.copy()
                if ENABLE_AMP:
                    model_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
                    # Usar AdamW para melhor estabilidade com AMP
                    if 'policy_kwargs' not in model_config:
                        model_config['policy_kwargs'] = {}
                    model_config['policy_kwargs']['optimizer_class'] = torch.optim.AdamW
                    model_config['policy_kwargs']['optimizer_kwargs'] = {
                        'eps': 1e-5,
                        'weight_decay': 0.01,
                        'amsgrad': False
                    }
                
                model = SAC(
                    SACTwoHeadPolicy,
                    env_train,
                    verbose=1,
                    tensorboard_log='training SAC/logs/tensorboard/',
                    **model_config
                )
                
                # 🔥 AMP: Configurar GradScaler para novo modelo
                if ENABLE_AMP and hasattr(model, 'policy'):
                    model._amp_scaler = GradScaler()
                    print("✅ GradScaler configurado para novo modelo")
                
                steps_already_trained = 0
        else:
            print("[INFO] Criando novo modelo SAC...")
            
            # 🔥 AMP: Configurações otimizadas para modelo novo
            model_config = BEST_PARAMS.copy()
            if ENABLE_AMP:
                model_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
                # Usar AdamW para melhor estabilidade com AMP
                if 'policy_kwargs' not in model_config:
                    model_config['policy_kwargs'] = {}
                model_config['policy_kwargs']['optimizer_class'] = torch.optim.AdamW
                model_config['policy_kwargs']['optimizer_kwargs'] = {
                    'eps': 1e-5,
                    'weight_decay': 0.01,
                    'amsgrad': False
                }
            
            model = SAC(
                SACTwoHeadPolicy,
                env_train,
                verbose=1,
                tensorboard_log='training SAC/logs/tensorboard/',
                **model_config
            )
            
            # 🔥 AMP: Configurar GradScaler para novo modelo
            if ENABLE_AMP and hasattr(model, 'policy'):
                model._amp_scaler = GradScaler()
                print("✅ GradScaler configurado para novo modelo")
                print("🚀 AMP ativado - Treinamento acelerado!")
        
        # 🎯 CALCULAR STEPS RESTANTES (2M TOTAL!)
        TOTAL_TIMESTEPS = 2000000  # 🚀 2 MILHÕES DE TIMESTEPS!
        remaining_steps = max(0, TOTAL_TIMESTEPS - steps_already_trained)
        
        print(f"\n🚀 INICIANDO FASE 1: Treinamento SAC ({remaining_steps//1000}k steps)")
        print(f"💾 Salvamento automático a cada 10k steps em: training SAC/checkpoints")
        
        if steps_already_trained > 0:
            print(f"🔄 RESUMINDO do step {steps_already_trained:,}")
            print(f"🎯 Steps restantes: {remaining_steps:,}")
        else:
            print(f"🚀 Treinamento completo: {remaining_steps:,} steps")
        
        print(f"📊 META TOTAL: {TOTAL_TIMESTEPS:,} steps (2x dataset completo!)")
        
        # 🔥 VERIFICAÇÃO FINAL ANTES DO TREINAMENTO
        print("\n🔍 VERIFICAÇÃO FINAL DO MODELO...")
        try:
            # Verificar se modelo está em modo train
            is_training = model.policy.training
            print(f"✅ Modo de treinamento: {'ATIVO' if is_training else '❌ INATIVO'}")
            
            # Verificar parâmetros treináveis
            trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.policy.parameters())
            print(f"✅ Parâmetros treináveis: {trainable_params:,}/{total_params:,}")
            
            # Verificar otimizador
            if hasattr(model.policy, 'optimizer'):
                lr = model.policy.optimizer.param_groups[0]['lr']
                print(f"✅ Learning Rate: {lr:.2e}")
            
            # Teste rápido de forward pass
            print("🧪 Testando forward pass...")
            obs = env_train.reset()
            if hasattr(obs, 'shape'):
                print(f"✅ Observação shape: {obs.shape}")
                # Teste de ação
                action, _ = model.predict(obs, deterministic=False)
                print(f"✅ Ação gerada: shape {action.shape}")
            
            print("✅ MODELO PRONTO PARA TREINAMENTO!")
            
        except Exception as e:
            print(f"⚠️ Erro na verificação final: {e}")
            print("🔄 Continuando mesmo assim...")
        
        start_time = time.time()
        
        # Callback com checkpointing automático
        class AdvancedMetricsCallback(BaseCallback):
            def __init__(self, log_freq=500, checkpoint_freq=10000):
                super().__init__()
                self.log_freq = log_freq  # Initialize log_freq attribute
                self.last_metrics_step = 0
                self.checkpoint_freq = checkpoint_freq
                self.last_weights_hash = None  # 🔥 NOVO: Para verificar mudanças nos pesos
                self.training_verification_freq = 100  # Verificar a cada 100 steps
                self.no_learning_count = 0  # Contador de steps sem aprendizado
            
            def on_training_start(self, locals, globals):
                """Called when training starts"""
                print("🚀 CALLBACK: Treinamento iniciado - verificando configuração...")
                
                # 🔥 VERIFICAÇÃO INICIAL DOS PESOS
                if hasattr(self.model.policy, 'state_dict'):
                    state_dict = self.model.policy.state_dict()
                    self.last_weights_hash = self._calculate_weights_hash(state_dict)
                    print(f"🔍 HASH INICIAL DOS PESOS: {self.last_weights_hash}")
                
                # 🔥 VERIFICAR CONFIGURAÇÃO DOS OTIMIZADORES SAC
                if hasattr(self.model.policy, 'actor') and hasattr(self.model.policy.actor, 'optimizer'):
                    actor_opt = self.model.policy.actor.optimizer
                    print(f"🔍 OTIMIZADOR ACTOR: {type(actor_opt).__name__ if actor_opt else 'None'}")
                    if actor_opt and hasattr(actor_opt, 'param_groups'):
                        print(f"🔍 ACTOR LR: {actor_opt.param_groups[0]['lr']}")
                
                if hasattr(self.model.policy, 'critic') and hasattr(self.model.policy.critic, 'optimizer'):
                    critic_opt = self.model.policy.critic.optimizer
                    print(f"🔍 OTIMIZADOR CRITIC: {type(critic_opt).__name__ if critic_opt else 'None'}")
                    if critic_opt and hasattr(critic_opt, 'param_groups'):
                        print(f"🔍 CRITIC LR: {critic_opt.param_groups[0]['lr']}")
                
                # Verificar otimizadores do SAC
                if hasattr(self.model, 'actor') and hasattr(self.model.actor, 'optimizer'):
                    print(f"🔍 SAC ACTOR OPTIMIZER: {type(self.model.actor.optimizer).__name__}")
                if hasattr(self.model, 'critic') and hasattr(self.model.critic, 'optimizer'):
                    print(f"🔍 SAC CRITIC OPTIMIZER: {type(self.model.critic.optimizer).__name__}")
                
                # 🔥 VERIFICAR GRADIENTES
                self._verify_gradients()
                
            def _calculate_weights_hash(self, state_dict):
                """Calcula hash dos pesos para detectar mudanças"""
                try:
                    # Concatenar todos os tensores e calcular hash
                    all_weights = torch.cat([p.flatten() for p in state_dict.values()])
                    return hash(all_weights.detach().cpu().numpy().tobytes())
                except Exception as e:
                    print(f"[WARNING] Erro ao calcular hash dos pesos: {e}")
                    return None
            
            def _verify_gradients(self):
                """Verificar se gradientes estão sendo calculados"""
                try:
                    if hasattr(self.model.policy, 'parameters'):
                        total_params = 0
                        params_with_grad = 0
                        
                        for param in self.model.policy.parameters():
                            total_params += 1
                            if param.grad is not None:
                                params_with_grad += 1
                        
                        print(f"🔍 GRADIENTES: {params_with_grad}/{total_params} parâmetros com gradientes")
                        
                        if params_with_grad == 0:
                            print("🚨 ERRO CRÍTICO: NENHUM PARÂMETRO TEM GRADIENTES!")
                            print("🚨 POSSÍVEIS CAUSAS:")
                            print("   - Ambiente não está retornando loss")
                            print("   - Modelo está em modo eval")
                            print("   - Problema na função de reward")
                        
                except Exception as e:
                    print(f"[WARNING] Erro na verificação de gradientes: {e}")
            
            def _on_step(self) -> bool:
                # 🔥 CORREÇÃO: Usar num_timesteps do modelo diretamente (já inclui progresso anterior)
                total_steps = self.num_timesteps
                
                # 🔥 VERIFICAÇÃO DE APRENDIZADO A CADA 2048 STEPS (após um batch completo)
                # 🚨 TEMPORARIAMENTE DESABILITADO - Causando encerramento precoce
                # if self.num_timesteps % 2048 == 0 and self.num_timesteps > 2048:
                #     if not self._verify_model_is_learning():
                #         self.no_learning_count += 1
                #         print(f"⚠️ STEP {total_steps}: Modelo não está aprendendo! Contador: {self.no_learning_count}")
                #         
                #         # 🔥 CORREÇÃO: Ser mais tolerante - 100 batches (200k steps) sem aprender
                #         if self.no_learning_count >= 100:  # Era 20, agora 100 batches
                #             print("🚨 ERRO CRÍTICO: Modelo não está aprendendo há 200k steps!")
                #             print("🚨 INTERROMPENDO TREINAMENTO PARA DIAGNÓSTICO...")
                #             return False  # Interromper treinamento
                #     else:
                #         self.no_learning_count = 0  # Reset contador se está aprendendo
                #         print(f"✅ STEP {total_steps}: Modelo está aprendendo! Contador resetado.")
                
                # 🔥 VERIFICAÇÃO SIMPLIFICADA: Apenas log de progresso sem interrupção
                if self.num_timesteps % 10000 == 0 and self.num_timesteps > 0:
                    if hasattr(self, '_verify_model_is_learning'):
                        is_learning = self._verify_model_is_learning()
                        status = "✅ APRENDENDO" if is_learning else "⚠️ ESTÁTICO"
                        print(f"📊 STEP {total_steps}: Status do modelo: {status}")
                    else:
                        print(f"📊 STEP {total_steps}: Treinamento em progresso...")
                
                # 🧪 CHECKPOINT DE TESTE NO PRIMEIRO STEP
                if self.num_timesteps == 1:
                    print(">>> 🧪 SALVANDO CHECKPOINT DE TESTE NO STEP 1 <<<")
                    self._save_auto_checkpoint(total_steps)
                
                # 🔍 VERIFICAÇÃO DE TREINAMENTO ATIVO (a cada 1000 steps)
                if self.num_timesteps % 1000 == 0:
                    self._verify_training_progress(total_steps)
                
                # Log a cada 500 steps
                if self.num_timesteps % self.log_freq == 0:
                    self._log_metrics(total_steps)
                
                # Checkpoint automático a cada 10k steps
                if self.num_timesteps % self.checkpoint_freq == 0:
                    self._save_auto_checkpoint(total_steps)
                
                return True
            
            def _verify_model_is_learning(self):
                """Verificar se o modelo está realmente aprendendo"""
                try:
                    learning_indicators = 0
                    total_indicators = 0
                    
                    # 1. Verificar mudanças nos pesos
                    if hasattr(self.model.policy, 'state_dict'):
                        state_dict = self.model.policy.state_dict()
                        current_hash = self._calculate_weights_hash(state_dict)
                        
                        if self.last_weights_hash is not None and current_hash is not None:
                            weights_changed = current_hash != self.last_weights_hash
                            if weights_changed:
                                learning_indicators += 1
                                self.last_weights_hash = current_hash
                            total_indicators += 1
                    
                    # 2. Verificar se há gradientes ativos
                    if hasattr(self.model.policy, 'parameters'):
                        params_with_grad = 0
                        total_params = 0
                        
                        for param in self.model.policy.parameters():
                            total_params += 1
                            if param.grad is not None and torch.any(param.grad != 0):
                                params_with_grad += 1
                        
                        if params_with_grad > 0:
                            learning_indicators += 1
                        total_indicators += 1
                    
                    # 3. Verificar se otimizador está ativo
                    if (hasattr(self.model.policy, 'optimizer') and 
                        self.model.policy.optimizer is not None and
                        hasattr(self.model.policy.optimizer, 'param_groups') and
                        len(self.model.policy.optimizer.param_groups) > 0):
                        # Verificar se learning rate é válido
                        lr = self.model.policy.optimizer.param_groups[0]['lr']
                        if lr > 0:
                            learning_indicators += 1
                        total_indicators += 1
                    
                    # Considerar que está aprendendo se pelo menos 1 indicador for positivo
                    is_learning = learning_indicators > 0 if total_indicators > 0 else True
                    
                    # Log detalhado apenas quando não está aprendendo
                    if not is_learning and total_indicators > 0:
                        print(f"[DEBUG] Indicadores de aprendizado: {learning_indicators}/{total_indicators}")
                    
                    return is_learning
                    
                except Exception as e:
                    print(f"[WARNING] Erro na verificação de aprendizado: {e}")
                    return True  # Assumir que está aprendendo se não conseguir verificar
            
            def _log_metrics(self, total_steps):
                """Log métricas de treinamento"""
                try:
                    # Obter learning rate atual
                    current_lr = 0.0
                    if (hasattr(self.model, 'policy') and 
                        hasattr(self.model.policy, 'optimizer') and
                        self.model.policy.optimizer is not None and
                        hasattr(self.model.policy.optimizer, 'param_groups')):
                        for param_group in self.model.policy.optimizer.param_groups:
                            current_lr = param_group['lr']
                            break
                    
                    # Log básico de progresso
                    progress_pct = (total_steps / 2000000) * 100
                    print(f"[STEP {total_steps:,}] Progresso: {progress_pct:.1f}% | LR: {current_lr:.2e}")
                    
                    # 🔥 NOVO: Verificar se modelo está aprendendo
                    if hasattr(self, 'last_weights_hash') and self.last_weights_hash is not None:
                        learning_status = "✅ APRENDENDO" if self.no_learning_count == 0 else f"⚠️ SEM APRENDER ({self.no_learning_count})"
                        print(f"[STEP {total_steps:,}] Status: {learning_status}")
                    
                    # Log detalhado a cada 5k steps
                    if total_steps % 5000 == 0:
                        print(f">>> CHECKPOINT PROGRESSO - Step {total_steps:,} <<<")
                        print(f"  📊 Progresso total: {progress_pct:.2f}%")
                        print(f"  🎯 Meta: 2,000,000 steps")
                        print(f"  ⏱️ Tempo decorrido: {(time.time() - start_time)/3600:.1f}h")
                        print(f"  🧠 Status aprendizado: {learning_status}")
                        
                except Exception as e:
                    print(f"[WARNING] Erro no log de métricas: {e}")
            
            def _verify_training_progress(self, total_steps):
                """Verificar se o treinamento está progredindo"""
                try:
                    if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'state_dict'):
                        # Verificar se os pesos estão sendo atualizados
                        state_dict = self.model.policy.state_dict()
                        total_params = sum(p.numel() for p in state_dict.values())
                        
                        if total_steps % 5000 == 0:  # Log detalhado a cada 5k
                            non_zero_params = sum(torch.count_nonzero(p).item() for p in state_dict.values())
                            zero_ratio = (total_params - non_zero_params) / total_params
                            print(f"[VERIFY] Parâmetros: {total_params:,} total, {non_zero_params:,} não-zero ({(1-zero_ratio)*100:.1f}%)")
                            
                            # 🔥 NOVO: Verificar mudanças nos pesos
                            current_hash = self._calculate_weights_hash(state_dict)
                            if hasattr(self, 'last_verification_hash') and self.last_verification_hash is not None:
                                weights_changed = current_hash != self.last_verification_hash
                                print(f"[VERIFY] Pesos mudaram desde última verificação: {'✅ SIM' if weights_changed else '❌ NÃO'}")
                            self.last_verification_hash = current_hash
                            
                except Exception as e:
                    print(f"[WARNING] Erro na verificação: {e}")
            
            def _save_auto_checkpoint(self, total_steps):
                """Salvar checkpoint automático"""
                try:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # 🔥 VERIFICAÇÃO PRÉ-SALVAMENTO: Verificar se modelo tem pesos válidos
                    if hasattr(self.model.policy, 'state_dict'):
                        state_dict = self.model.policy.state_dict()
                        total_params = sum(p.numel() for p in state_dict.values())
                        non_zero_params = sum(torch.count_nonzero(p).item() for p in state_dict.values())
                        zero_ratio = (total_params - non_zero_params) / total_params
                        
                        print(f"🔍 PRÉ-SALVAMENTO: {total_params:,} parâmetros, {non_zero_params:,} não-zero ({(1-zero_ratio)*100:.1f}%)")
                        
                        if zero_ratio > 0.8:
                            print("🚨 AVISO: Modelo tem muitos parâmetros zero - pode não estar treinando!")
                    
                    # 🔥 SALVAMENTO ROBUSTO EM MÚLTIPLOS LOCAIS
                    # 1. Root directory (compatibilidade com mainppo1.py)
                    root_path = f"auto_checkpoint_{total_steps}_steps.zip"
                    
                    # 2. Framework directory
                    sac_dir = "training SAC/checkpoints"
                    os.makedirs(sac_dir, exist_ok=True)
                    sac_path = f"{sac_dir}/checkpoint_{total_steps}_steps_{timestamp}.zip"
                    
                    print(f">>> 💾 SALVANDO CHECKPOINT AUTOMÁTICO - Step {total_steps:,} <<<")
                    
                    # Salvar no root (compatibilidade)
                    print(f"💾 Salvando em: {root_path}")
                    self.model.save(root_path)
                    
                    # Salvar no training SAC
                    print(f"💾 Salvando em: {sac_path}")
                    self.model.save(sac_path)
                    print("✅ model.save() executado")
                    
                    # 🔥 VERIFICAÇÃO PÓS-SALVAMENTO COMPLETA
                    if os.path.exists(sac_path):
                        size_bytes = os.path.getsize(sac_path)
                        size_mb = size_bytes / (1024*1024)
                        print(f"✅ Arquivo criado: {size_mb:.1f}MB ({size_bytes:,} bytes)")
                    
                        # 🔥 VERIFICAÇÃO DE TAMANHO CRÍTICA
                        if size_mb < 5:
                            print(f"🚨 ERRO CRÍTICO: Modelo muito pequeno ({size_mb:.1f}MB)!")
                            print("🚨 POSSÍVEIS CAUSAS:")
                            print("   - Modelo não foi treinado")
                            print("   - Erro no salvamento")
                            print("   - Pesos não foram atualizados")
                            
                            # Tentar salvar novamente com nome diferente
                            backup_path = f"{sac_dir}/EMERGENCY_sac_{total_steps}_{timestamp}.zip"
                            print(f"🆘 Tentando salvamento de emergência: {backup_path}")
                            self.model.save(backup_path)
                    
                        elif size_mb > 50:
                            print(f"⚠️ AVISO: Modelo muito grande ({size_mb:.1f}MB) - verificar se normal")
                        else:
                            print(f"🎯 TAMANHO NORMAL: {size_mb:.1f}MB - modelo válido!")
                            
                        # 🔥 TESTE DE CARREGAMENTO RÁPIDO
                        try:
                            print("🧪 Testando carregamento do checkpoint...")
                            test_model = SAC.load(sac_path, env=None)
                            if test_model is not None:
                                print("✅ Checkpoint pode ser carregado corretamente!")
                                del test_model  # Liberar memória
                            else:
                                print("❌ ERRO: Checkpoint não pode ser carregado!")
                        except Exception as load_error:
                            print(f"❌ ERRO no teste de carregamento: {load_error}")
                            
                    else:
                        print("❌ ERRO CRÍTICO: Arquivo não foi criado!")
                        
                    print(f">>> 💾 CHECKPOINT COMPLETO: {sac_path} <<<\n")
        
                except Exception as e:
                    print(f"❌ ERRO CRÍTICO ao salvar checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 🆘 SALVAMENTO DE EMERGÊNCIA
                    try:
                        emergency_path = f"training SAC/EMERGENCY_SAVE_{total_steps}.zip"
                        print(f"🆘 Tentando salvamento de emergência: {emergency_path}")
                        self.model.save(emergency_path)
                        print("🆘 Salvamento de emergência concluído")
                    except Exception as emergency_error:
                        print(f"🆘 Falha no salvamento de emergência: {emergency_error}")
        
        # 🔥 CONTINUAÇÃO DE TREINAMENTO: Callback sem offset (modelo já preserva progresso)
        callback = AdvancedMetricsCallback(
            log_freq=500, 
            checkpoint_freq=10000
        )
        
        # 🔥 CORREÇÃO CRÍTICA: reset_num_timesteps baseado em continuação
        reset_timesteps = steps_already_trained == 0  # Só resetar se for modelo novo
        
        if steps_already_trained > 0:
            print(f">>> CONTINUANDO TREINAMENTO - reset_num_timesteps=False <<<")
        else:
            print(f">>> INICIANDO TREINAMENTO NOVO - reset_num_timesteps=True <<<")
        
        # 🚀 EXECUTAR TREINAMENTO
        try:
            model.learn(
                total_timesteps=remaining_steps,
                reset_num_timesteps=reset_timesteps,  # 🔥 CRÍTICO: Não resetar se continuar
                callback=callback,
                progress_bar=True
            )
            
            print(f"\n🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
            print(f"⏱️ Tempo total: {(time.time() - start_time)/3600:.1f} horas")
            
        except KeyboardInterrupt:
            print(f"\n⚠️ TREINAMENTO INTERROMPIDO PELO USUÁRIO")
            print(f"⏱️ Tempo decorrido: {(time.time() - start_time)/3600:.1f} horas")
            print(f"📊 Steps executados: {model.num_timesteps:,}")
            
        except Exception as e:
            print(f"\n❌ ERRO DURANTE TREINAMENTO: {e}")
            print(f"⏱️ Tempo decorrido: {(time.time() - start_time)/3600:.1f} horas")
            print(f"📊 Steps executados: {model.num_timesteps:,}")
        
        # 🏗️ SAC TRAINER: SALVAR MODELO FINAL ROBUSTO EM training SAC/
        try:
            # 🔥 CORREÇÃO: num_timesteps já inclui todo o progresso
            final_steps = model.num_timesteps
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_model_path = f"training SAC/sac_FINAL_{final_steps}_steps_{timestamp}.zip"
            os.makedirs("training SAC", exist_ok=True)
            
            print(f"\n>>> 💾 SALVANDO MODELO FINAL ROBUSTO <<<")
            
            # 🔥 VERIFICAÇÃO PRÉ-SALVAMENTO FINAL
            if hasattr(model.policy, 'state_dict'):
                state_dict = model.policy.state_dict()
                total_params = sum(p.numel() for p in state_dict.values())
                non_zero_params = sum(torch.count_nonzero(p).item() for p in state_dict.values())
                zero_ratio = (total_params - non_zero_params) / total_params
                print(f"🔍 MODELO FINAL: {total_params:,} parâmetros, {non_zero_params:,} não-zero ({(1-zero_ratio)*100:.1f}%)")
            
            model.save(final_model_path)
            
            # 🔥 VERIFICAÇÃO FINAL ROBUSTA
            if os.path.exists(final_model_path):
                size_bytes = os.path.getsize(final_model_path)
                size_mb = size_bytes / (1024*1024)
                
                print(f"\n✅ MODELO FINAL SALVO COM SUCESSO:")
                print(f"   📁 Location: {final_model_path}")
                print(f"   📊 Total steps: {final_steps:,}")
                print(f"   💾 Tamanho: {size_mb:.1f}MB ({size_bytes:,} bytes)")
                
                if size_mb < 5:
                    print(f"🚨 AVISO: Modelo final muito pequeno ({size_mb:.1f}MB)!")
                    # Salvar backup de emergência
                    emergency_final = f"training SAC/EMERGENCY_FINAL_{final_steps}_{timestamp}.zip"
                    model.save(emergency_final)
                    print(f"🆘 Backup de emergência salvo: {emergency_final}")
                elif size_mb > 50:
                    print(f"⚠️ AVISO: Modelo final muito grande ({size_mb:.1f}MB)")
                else:
                    print(f"🎯 TAMANHO PERFEITO: Modelo final válido!")
                    
                # Teste de carregamento final
                try:
                    print("🧪 Testando carregamento do modelo final...")
                    test_final = SAC.load(final_model_path, env=None)
                    if test_final is not None:
                        print("✅ Modelo final pode ser carregado corretamente!")
                        del test_final
                    else:
                        print("❌ ERRO: Modelo final não pode ser carregado!")
                except Exception as load_error:
                    print(f"❌ ERRO no teste de carregamento final: {load_error}")
            else:
                print("❌ ERRO CRÍTICO: Modelo final não foi criado!")
            
        except Exception as save_error:
            print(f"❌ ERRO CRÍTICO ao salvar modelo final: {save_error}")
            import traceback
            traceback.print_exc()
            
            # 🆘 SALVAMENTO DE EMERGÊNCIA FINAL
            try:
                emergency_final = f"training SAC/EMERGENCY_FINAL_{int(time.time())}.zip"
                print(f"🆘 Tentando salvamento final de emergência: {emergency_final}")
                model.save(emergency_final)
                print("🆘 Salvamento final de emergência concluído")
            except Exception as emergency_error:
                print(f"🆘 Falha no salvamento final de emergência: {emergency_error}")
        
        # 🔥 SALVAR VecNormalize se usado
        if USE_VECNORM and hasattr(env_train, 'save'):
            try:
                vec_normalize_path = 'vec_normalize_final.pkl'
                env_train.save(vec_normalize_path)
                print(f"✅ VecNormalize salvo: {vec_normalize_path}")
            except Exception as vec_error:
                print(f"⚠️ Erro ao salvar VecNormalize: {vec_error}")
        
        print(f"\nTreinamento concluido com sucesso!")
        print(f"\n{'='*60}")
        print(f"🏁 SAC TRAINING FINALIZADO")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n💥 ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    main()