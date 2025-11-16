# -*- coding: utf-8 -*-
"""
🚀 TREINAMENTO DIFERENCIADO PPO - SISTEMA AVANÇADO
Sistema especializado para treinamento PPO com configurações otimizadas

🎯 CARACTERÍSTICAS PRINCIPAIS:
- Configuração 1500 steps (5.2 dias por episódio)
- Day Trading otimizado
- Alvos realistas (10-50 pontos)
- Sistema de recompensas diferenciado
- SL/TP realistas (8-60 SL, 12-80 TP)

Autor: Assistant
Data: 2024
"""

# 🔥 CONFIGURAÇÕES GPU/CPU OTIMIZADAS DO MAINPPO1.PY
import torch
import multiprocessing
import os

# 🚀 CONFIGURAÇÕES PYTORCH OTIMIZADAS PARA RTX 4070ti
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🚀 GPU DETECTADA: {device_name}")
    print(f"💾 VRAM Total: {memory_total:.1f}GB")
    
    # 🎯 CONFIGURAÇÕES ESPECÍFICAS PARA RTX 4070ti
    if "4070" in device_name or memory_total >= 11.5:  # RTX 4070ti tem 12GB
        print("🎯 RTX 4070ti DETECTADA - Aplicando configurações OTIMIZADAS!")
        
        # Configurações agressivas para RTX 4070ti (Ada Lovelace)
        torch.backends.cudnn.benchmark = True  # Performance over reproducibility
        torch.backends.cudnn.allow_tf32 = True  # 1.7x speedup
        torch.backends.cuda.matmul.allow_tf32 = True  # 1.7x speedup
        torch.backends.cudnn.deterministic = False  # Performance over reproducibility
        torch.backends.cudnn.enabled = True
        
        # Configurações de memória específicas para 12GB
        torch.backends.cuda.max_split_size_mb = 1024  # 4070ti pode usar fragmentos maiores
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,roundup_power2_divisions:8"
        
        # Configurações avançadas para Ada Lovelace
        torch.backends.cuda.enable_math_sdp(True)  # Scaled Dot Product Attention otimizado
        torch.backends.cuda.enable_flash_sdp(True)  # Flash Attention se disponível
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory efficient attention
        
        # Configurar cache de kernel para Ada Lovelace
        os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"  # 2GB cache
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async launches
        
        print("✅ CONFIGURAÇÕES RTX 4070ti:")
        print("   🔥 TF32 ativado (1.7x speedup)")
        print("   ⚡ Flash Attention ativado")
        print("   💾 Fragmentação otimizada: 1024MB")
        print("   🚀 Kernel cache: 2GB")
        
    elif memory_total >= 7.5:  # RTX 4070 ou similar (8GB+)
        print("🎯 GPU de 8GB+ detectada - Configurações equilibradas")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.max_split_size_mb = 512
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
    else:  # GPUs menores
        print("⚠️ GPU <8GB detectada - Configurações conservadoras")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.max_split_size_mb = 256
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    
    # Limpar cache e configurar para treinamento
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 🔥 CONFIGURAÇÕES CPU OTIMIZADAS PARA RTX 4070ti
    cpu_cores = max(2, int(multiprocessing.cpu_count() * 0.75))  # 75% dos cores
    torch.set_num_threads(cpu_cores)  # Threads otimizadas para GPU
    torch.set_num_interop_threads(2)  # Fixo em 2 para evitar overhead
    
    # Configurações específicas para Stable Baselines3 + GPU
    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
    os.environ["MKL_NUM_THREADS"] = str(cpu_cores) 
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_cores)
    
    print(f"   🧮 CPU otimizada: {cpu_cores} threads ({multiprocessing.cpu_count() * 0.75:.0f}% dos cores)")
    
    print(f"🔧 CONFIGURAÇÕES FINAIS:")
    print(f"   CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   Max Split Size: {torch.backends.cuda.max_split_size_mb}MB")
    print(f"   CPU Threads: {torch.get_num_threads()}")
    print("=" * 60)
    
else:
    print("❌ GPU não disponível - usando CPU")
    # Configurações CPU otimizadas como fallback
    cpu_cores = max(2, int(multiprocessing.cpu_count() * 0.75))
    torch.set_num_threads(cpu_cores)
    torch.set_num_interop_threads(2)
    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
    os.environ["MKL_NUM_THREADS"] = str(cpu_cores)
    print(f"🔧 CPU configurado: {cpu_cores} threads")

# 🏗️ AMBIENTE MODULAR - IMPORTS ESSENCIAIS
import sys
import os
import numpy as np
import pandas as pd
import random
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
from gym import spaces
import logging
from datetime import datetime
import ta
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings
import glob
import psutil
import gc
import time
import threading
from queue import Queue
import json
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dataclasses import dataclass
from enum import Enum
import traceback
from collections import deque
from tqdm import tqdm

# 🎯 NOVO SISTEMA DE REWARDS DIFERENCIADO
from reward_system_diff import create_diff_reward_system, DIFF_REWARD_CONFIG
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor

# Caminhos exclusivos para DIFF
DIFF_MODEL_DIR = "Otimizacao/treino_principal/models/DIFF"
DIFF_CHECKPOINT_DIR = "Otimizacao/treino_principal/checkpoints/DIFF"
DIFF_ENVSTATE_DIR = "trading_framework/training/checkpoints/DIFF"

os.makedirs(DIFF_MODEL_DIR, exist_ok=True)
os.makedirs(DIFF_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DIFF_ENVSTATE_DIR, exist_ok=True)

# === FUNÇÕES DE CARREGAMENTO OTIMIZADO DE DADOS ===
def load_optimized_data():
    """
    🎯 CARREGAR DATASET GOLD FINAL SEM PREÇOS ESTÁTICOS - ESTRUTURA MAINPPO1.PY
    """
    try:
        # 🎯 PRIORIDADE 1: Dataset GOLD final sem preços estáticos
        gold_final_cache = "novos datasets/GOLD_final_nostatic.pkl"
        
        if os.path.exists(gold_final_cache):
            print(f"[OPTIMIZED CACHE] 🎯 Carregando dataset GOLD FINAL sem preços estáticos...")
            start_time = time.time()
            
            df = pd.read_pickle(gold_final_cache)
            
            load_time = time.time() - start_time
            print(f"[OPTIMIZED CACHE] ✅ Dataset GOLD FINAL carregado: {len(df):,} barras")
            print(f"[OPTIMIZED CACHE] 📅 Período: {df.index[0]} até {df.index[-1]}")
            print(f"[OPTIMIZED CACHE] ⏱️ Duração: {(df.index[-1] - df.index[0]).days} dias")
            print(f"[OPTIMIZED CACHE] ⚡ Tempo: {load_time:.3f}s")
            
            # 🎯 USAR DATASET COMPLETO - SPLIT FEITO NO TRADINGENV (igual mainppo1.py)
            print(f"[OPTIMIZED CACHE] 🎯 CONFIGURAÇÃO FINAL:")
            print(f"[OPTIMIZED CACHE]    🎯 Dataset final: {len(df):,} barras (100% sem preços estáticos)")
            print(f"[OPTIMIZED CACHE]    🎯 Split 80/16 será aplicado no TradingEnv (cortar primeiros 20%)")
            
            return df
        
        # 🎯 PRIORIDADE 2: Cache GOLD combinado otimizado
        gold_combined_cache = "novos datasets/GOLD_combined_optimized.pkl"
        
        if os.path.exists(gold_combined_cache):
            print(f"[OPTIMIZED CACHE] 🎯 Carregando dataset GOLD combinado otimizado...")
            start_time = time.time()
            
            df = pd.read_pickle(gold_combined_cache)
            
            load_time = time.time() - start_time
            print(f"[OPTIMIZED CACHE] ✅ Dataset GOLD carregado: {len(df):,} barras")
            print(f"[OPTIMIZED CACHE] 📅 Período: {df.index[0]} até {df.index[-1]}")
            print(f"[OPTIMIZED CACHE] ⏱️ Duração: {(df.index[-1] - df.index[0]).days} dias")
            print(f"[OPTIMIZED CACHE] ⚡ Tempo: {load_time:.3f}s")
            
            # 🎯 USAR DATASET COMPLETO - SPLIT FEITO NO TRADINGENV (igual mainppo1.py)
            print(f"[OPTIMIZED CACHE] 🎯 CONFIGURAÇÃO FINAL:")
            print(f"[OPTIMIZED CACHE]    🎯 Dataset completo: {len(df):,} barras")
            print(f"[OPTIMIZED CACHE]    🎯 Split 80/16 será aplicado no TradingEnv (cortar primeiros 20%)")
            
            return df
        
        # 🎯 PRIORIDADE 3: Cache GOLD individual otimizado
        gold_train_cache = "novos datasets/GOLD_train_optimized.pkl"
        gold_val_cache = "novos datasets/GOLD_val_optimized.pkl"
        
        if os.path.exists(gold_train_cache) and os.path.exists(gold_val_cache):
            print(f"[OPTIMIZED CACHE] 🎯 Carregando datasets GOLD individuais otimizados...")
            start_time = time.time()
            
            df_train = pd.read_pickle(gold_train_cache)
            df_val = pd.read_pickle(gold_val_cache)
            
            # 🎯 COMBINAR DATASETS CRONOLOGICAMENTE - SPLIT FEITO NO TRADINGENV (igual mainppo1.py)
            df = pd.concat([df_train, df_val]).sort_index()
            
            load_time = time.time() - start_time
            print(f"[OPTIMIZED CACHE] ✅ Dataset GOLD combinado: {len(df):,} barras")
            print(f"[OPTIMIZED CACHE] 📅 Período: {df.index[0]} até {df.index[-1]}")
            print(f"[OPTIMIZED CACHE] ⏱️ Duração: {(df.index[-1] - df.index[0]).days} dias")
            print(f"[OPTIMIZED CACHE] ⚡ Tempo: {load_time:.3f}s")
            
            return df
        
        # 🎯 PRIORIDADE 4: Cache específico do DIFF
        diff_cache = "data_cache/treinodiff_optimized_dataset.pkl"
        
        if os.path.exists(diff_cache):
            print(f"[OPTIMIZED CACHE] 🎯 Usando cache específico do DIFF...")
            start_time = time.time()
            
            df = pd.read_pickle(diff_cache)
            
            load_time = time.time() - start_time
            print(f"[OPTIMIZED CACHE] ✅ Cache DIFF carregado: {len(df):,} barras")
            print(f"[OPTIMIZED CACHE] ⚡ Tempo: {load_time:.3f}s")
            
            return df
        
        # 🎯 FALLBACK: Dataset original
        print(f"[OPTIMIZED CACHE] Cache não encontrado, usando fallback...")
        return get_latest_processed_file_fallback()
        
    except Exception as e:
        print(f"[OPTIMIZED CACHE] ❌ Erro ao carregar cache: {e}")
        print(f"[OPTIMIZED CACHE] Usando fallback...")
        return get_latest_processed_file_fallback()

def get_latest_processed_file_fallback():
    """
    🎯 CARREGAMENTO ROBUSTO DE DATASET COM FALLBACKS MÚLTIPLOS (FALLBACK)
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
            '5m': 'data/GOLD_5m_2025058_12580.csv',
            '15m': 'data/GOLD_15m_2025058_12580.csv', 
            '4h': 'data/GOLD_4h_2025058_12580.csv'
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

def create_synthetic_dataset():
    """
    🎯 CRIAR DATASET SINTÉTICO PARA TESTES DE EMERGÊNCIA
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
            # Simular OHLC baseado no preço base
            high = prices * (1 + np.random.uniform(0, 0.002, n_bars))
            low = prices * (1 - np.random.uniform(0, 0.002, n_bars))
            open_price = prices * (1 + np.random.uniform(-0.001, 0.001, n_bars))
            close_price = prices * (1 + np.random.uniform(-0.001, 0.001, n_bars))
            volume = np.random.uniform(100, 1000, n_bars)
            
            data[f'open_{tf}'] = open_price
            data[f'high_{tf}'] = high
            data[f'low_{tf}'] = low
            data[f'close_{tf}'] = close_price
            data[f'volume_{tf}'] = volume
        
        df = pd.DataFrame(data, index=dates)
        print(f"[SYNTHETIC] ✅ Dataset sintético criado: {len(df):,} barras")
        return df
        
    except Exception as e:
        print(f"[ERROR] Erro ao criar dataset sintético: {e}")
        # Criar dataset mínimo
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='5T')
        df = pd.DataFrame({
            'close_5m': np.random.uniform(2000, 2100, 1000),
            'volume_5m': np.random.uniform(100, 1000, 1000)
        }, index=dates)
        print(f"[SYNTHETIC] Dataset mínimo criado: {len(df)} barras")
        return df

# === CLASSES E SISTEMAS AVANÇADOS ===
class AdvancedMetricsSystem:
    """
    🎯 SISTEMA AVANÇADO DE MÉTRICAS EM TEMPO REAL
    """
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.portfolio_values = deque(maxlen=window_size)
        self.returns = deque(maxlen=window_size)
        self.drawdowns = deque(maxlen=window_size)
        self.trades = deque(maxlen=window_size)
        self.current_step = 0
        
    def update(self, portfolio_value, returns, drawdown, trades, current_step):
        """Atualizar métricas em tempo real"""
        self.portfolio_values.append(portfolio_value)
        self.returns.append(returns)
        self.drawdowns.append(drawdown)
        self.trades.append(trades)
        self.current_step = current_step
        
    def _calculate_advanced_metrics(self, portfolio_value, trades, current_step):
        """Calcular métricas avançadas"""
        if len(self.portfolio_values) < 2:
            return {}
            
        # Retorno total
        total_return = (portfolio_value - 1000) / 1000 * 100
        
        # Retorno médio por trade
        if trades:
            avg_trade_return = total_return / len(trades)
        else:
            avg_trade_return = 0
            
        # Volatilidade dos retornos
        if len(self.returns) > 1:
            volatility = np.std(list(self.returns)) * 100
        else:
            volatility = 0
            
        # Sharpe ratio simplificado
        if volatility > 0:
            sharpe_ratio = (np.mean(list(self.returns)) * 100) / volatility
        else:
            sharpe_ratio = 0
            
        # Drawdown máximo
        max_drawdown = max(list(self.drawdowns)) if self.drawdowns else 0
        
        # Win rate
        if trades:
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            win_rate = (winning_trades / len(trades)) * 100
        else:
            win_rate = 0
            
        return {
            'total_return': total_return,
            'avg_trade_return': avg_trade_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'current_step': current_step
        }
        
    def get_real_time_summary(self):
        """Obter resumo em tempo real"""
        if not self.portfolio_values:
            return "Aguardando dados..."
            
        current_value = self.portfolio_values[-1]
        metrics = self._calculate_advanced_metrics(current_value, list(self.trades), self.current_step)
        
        summary = f"""
🎯 MÉTRICAS EM TEMPO REAL:
💰 Portfolio: ${current_value:.2f}
📈 Retorno Total: {metrics['total_return']:.2f}%
📊 Volatilidade: {metrics['volatility']:.2f}%
📉 Drawdown Máx: {metrics['max_drawdown']:.2f}%
🎯 Win Rate: {metrics['win_rate']:.1f}%
📋 Total Trades: {metrics['total_trades']}
⏱️ Step: {self.current_step}
        """
        return summary
        
    def get_summary(self):
        """Obter resumo completo"""
        return self.get_real_time_summary()

# === AMBIENTE DE TRADING ===
# REMOVIDO: TradingEnv local - usando FrameworkTradingEnv unificado
class PhaseType(Enum):
    FUNDAMENTALS = "fundamentals"
    RISK_MANAGEMENT = "risk_management" 
    NOISE_HANDLING = "noise_handling"
    STRESS_TESTING = "stress_testing"
    INTEGRATION = "integration"

@dataclass
class TrainingPhase:
    name: str
    phase_type: PhaseType
    timesteps: int
    description: str
    data_filter: str
    success_criteria: Dict[str, float]
    reset_criteria: Dict[str, float]
    evaluation_freq: int = 10000

class PhaseMetrics:
    def __init__(self):
        self.phase_data = {}
    
    def add_metrics(self, phase: str, metrics: Dict):
        if phase not in self.phase_data:
            self.phase_data[phase] = []
        self.phase_data[phase].append({
            'timestamp': time.time(),
            'metrics': metrics
        })
    
    def get_phase_progress(self, phase: str) -> List[Dict]:
        return self.phase_data.get(phase, [])
    
    def is_plateauing(self, phase: str, window: int = 5) -> bool:
        progress = self.get_phase_progress(phase)
        if len(progress) < window:
            return False
        
        recent_metrics = progress[-window:]
        sharpe_ratios = [m['metrics'].get('sharpe_ratio', 0) for m in recent_metrics]
        
        # Verificar se há melhoria significativa
        if len(sharpe_ratios) >= 2:
            improvement = sharpe_ratios[-1] - sharpe_ratios[0]
            return improvement < 0.01  # Plateau se melhoria < 0.01
        
        return False
    
    def is_degrading(self, phase: str, window: int = 3) -> bool:
        progress = self.get_phase_progress(phase)
        if len(progress) < window:
            return False
        
        recent_metrics = progress[-window:]
        sharpe_ratios = [m['metrics'].get('sharpe_ratio', 0) for m in recent_metrics]
        
        # Verificar se está piorando consistentemente
        if len(sharpe_ratios) >= 2:
            degradation = sharpe_ratios[0] - sharpe_ratios[-1]
            return degradation > 0.05  # Degradação se piorou > 0.05
        
        return False

class AdaptiveReset:
    def __init__(self):
        self.reset_history = []
    
    def should_reset(self, phase: TrainingPhase, current_metrics: Dict) -> Tuple[bool, str]:
        # Verificar critérios de reset
        for criterion, threshold in phase.reset_criteria.items():
            current_value = current_metrics.get(criterion, 0)
            
            if criterion == "win_rate" and current_value < threshold:
                return True, f"Win rate muito baixo: {current_value:.3f} < {threshold:.3f}"
            
            elif criterion == "max_drawdown" and current_value > threshold:
                return True, f"Drawdown muito alto: {current_value:.3f} > {threshold:.3f}"
            
            elif criterion == "sharpe_ratio" and current_value < threshold:
                return True, f"Sharpe ratio muito baixo: {current_value:.3f} < {threshold:.3f}"
        
        return False, ""

class RobustSaveCallback(BaseCallback):
    def __init__(self, save_freq=10000, save_path="Otimizacao/treino_principal/models", name_prefix="model", total_steps_offset=0, training_env=None):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.total_steps_offset = total_steps_offset  # 🔥 NOVO: Offset para steps acumulados
        self.training_env = training_env  # 🔥 CORREÇÃO: Passar environment via parâmetro
        os.makedirs(save_path, exist_ok=True)
        # 🔥 NOVO: Criar diretórios de emergência
        os.makedirs("trading_framework/training", exist_ok=True)
        os.makedirs("trading_framework/training/checkpoints", exist_ok=True)
        print(f"🔧 RobustSaveCallback inicializado: save_freq={save_freq}, offset={total_steps_offset}")
        
    def _on_step(self) -> bool:
        # 🔥 CORREÇÃO: Usar steps acumulados reais para decidir quando salvar
        real_timesteps = self.num_timesteps + self.total_steps_offset
        
        # 🔥 NOVO: Salvar a cada múltiplo exato de save_freq
        if real_timesteps > 0 and real_timesteps % self.save_freq == 0:
            print(f"\n🎯 TRIGGER DE SALVAMENTO: Step {real_timesteps:,} (múltiplo de {self.save_freq:,})")
            try:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 🔥 SALVAMENTO ROBUSTO EM MÚLTIPLOS LOCAIS
                # 1. Root directory (compatibilidade com massive.py)
                root_path = f"auto_checkpoint_{real_timesteps}_steps.zip"
                
                # 2. Framework directory
                framework_dir = "trading_framework/training/checkpoints"
                framework_path = f"{framework_dir}/checkpoint_{real_timesteps}_steps_{timestamp}.zip"
                
                # 3. Original save path
                model_path = f"{self.save_path}/{self.name_prefix}_{real_timesteps}_steps_{timestamp}.zip"
                
                print(f"\n>>> 💾 SALVANDO CHECKPOINT ROBUSTO - Step {real_timesteps:,} <<<")
                print(f"    📊 Atual: {self.num_timesteps:,} + Offset: {self.total_steps_offset:,} = {real_timesteps:,}")
                
                # 🔥 CORREÇÃO: Salvar estado do ambiente usando self.training_env
                env_state = {}
                if self.training_env is not None:
                    try:
                        # Acessar ambiente interno corretamente
                        actual_env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0 else self.training_env
                        
                        env_state = {
                            'current_step': getattr(actual_env, 'current_step', 0),
                            'episode_steps': getattr(actual_env, 'episode_steps', 0),
                            'portfolio_value': getattr(actual_env, 'portfolio_value', 1000),
                            'realized_balance': getattr(actual_env, 'realized_balance', 1000),
                            'peak_portfolio': getattr(actual_env, 'peak_portfolio', 1000),
                            'positions': getattr(actual_env, 'positions', []),
                            'trades': getattr(actual_env, 'trades', []),
                            'current_drawdown': getattr(actual_env, 'current_drawdown', 0.0),
                            'peak_drawdown': getattr(actual_env, 'peak_drawdown', 0.0),
                            'win_streak': getattr(actual_env, 'win_streak', 0),
                            'steps_since_last_trade': getattr(actual_env, 'steps_since_last_trade', 0),
                            'total_timesteps': real_timesteps
                        }
                        
                        # Salvar estado do ambiente
                        env_state_path = f"{framework_dir}/env_state_{real_timesteps}_steps_{timestamp}.json"
                        with open(env_state_path, 'w') as f:
                            import json
                            json.dump(env_state, f, indent=2, default=str)
                        print(f"💾 Estado do ambiente salvo: {env_state_path}")
                        
                    except Exception as env_error:
                        print(f"⚠️ Erro ao salvar estado do ambiente: {env_error}")
                
                # 🔥 SALVAMENTO ROBUSTO COM VERIFICAÇÃO COMPLETA
                save_paths = [
                    ("Root", root_path),
                    ("Framework", framework_path), 
                    ("Original", model_path)
                ]
                
                successful_saves = 0
                for path_name, path in save_paths:
                    try:
                        print(f"💾 Salvando {path_name}: {path}")
                        
                        # 🔥 GARANTIR DIRETÓRIO EXISTE (só se não for root)
                        if os.path.dirname(path):  # Só se não for string vazia
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                        
                        # 🔥 SALVAMENTO SIMPLES COMO NO MAINPPO1-OLD2.PY
                        self.model.save(path)
                        
                        # Verificar se foi salvo corretamente
                        if os.path.exists(path):
                            size_bytes = os.path.getsize(path)
                            size_mb = size_bytes / (1024*1024)
                            print(f"✅ {path_name}: {size_mb:.1f}MB")
                            
                            # 🔥 VERIFICAÇÃO DETALHADA DO CONTEÚDO
                            try:
                                import zipfile
                                with zipfile.ZipFile(path, 'r') as z:
                                    files_in_zip = [f.filename for f in z.filelist]
                                    has_policy = any('policy.pth' in f for f in files_in_zip)
                                    has_pytorch_vars = any('pytorch_variables.pth' in f for f in files_in_zip)
                                    
                                    if has_policy and has_pytorch_vars:
                                        successful_saves += 1
                                        print(f"🎯 {path_name}: Salvamento COMPLETO (policy.pth + pytorch_variables.pth)!")
                                    else:
                                        print(f"⚠️ {path_name}: Salvamento INCOMPLETO - Faltam arquivos essenciais")
                                        print(f"    📁 Arquivos no ZIP: {files_in_zip}")
                            except Exception as zip_error:
                                print(f"⚠️ {path_name}: Erro ao verificar ZIP: {zip_error}")
                                if size_mb > 2.5:  # Se for grande, assumir que está OK
                                    successful_saves += 1
                                    print(f"🎯 {path_name}: Assumindo válido pelo tamanho ({size_mb:.1f}MB)")
                        else:
                            print(f"❌ {path_name}: Arquivo não foi criado!")
                            
                    except Exception as save_error:
                        print(f"❌ Erro ao salvar {path_name}: {save_error}")
                        import traceback
                        traceback.print_exc()
                
                print(f"📊 Salvamentos bem-sucedidos: {successful_saves}/3")
                
                # 🆘 SISTEMA DE EMERGÊNCIA MELHORADO + SALVAMENTO ALTERNATIVO
                if successful_saves == 0:
                    print("🚨 NENHUM SALVAMENTO FUNCIONOU - ATIVANDO EMERGÊNCIA TOTAL!")
                    
                    # 🔥 MÉTODO ALTERNATIVO: Salvamento manual dos componentes
                    try:
                        print("🔧 Tentando salvamento manual dos componentes...")
                        manual_path = f"MANUAL_SAVE_{real_timesteps}_{timestamp}"
                        os.makedirs(manual_path, exist_ok=True)
                        
                        # Salvar componentes individuais
                        import torch
                        if hasattr(self.model, 'policy'):
                            torch.save(self.model.policy.state_dict(), f"{manual_path}/policy.pth")
                            print("✅ policy.pth salvo manualmente")
                        
                        # 🔥 Salvamento de optimizer removido
                        
                        # Criar ZIP manual
                        import zipfile
                        manual_zip = f"{manual_path}.zip"
                        with zipfile.ZipFile(manual_zip, 'w') as z:
                            for file in os.listdir(manual_path):
                                z.write(f"{manual_path}/{file}", file)
                        
                        if os.path.exists(manual_zip):
                            size_mb = os.path.getsize(manual_zip) / (1024*1024)
                            print(f"🔧 Salvamento manual criado: {manual_zip} ({size_mb:.1f}MB)")
                            successful_saves += 1
                            
                    except Exception as manual_error:
                        print(f"🔧 Salvamento manual falhou: {manual_error}")
                    
                    # Emergências tradicionais
                    emergency_paths = [
                        f"EMERGENCY_CRITICAL_{real_timesteps}_{timestamp}.zip",
                        f"trading_framework/training/EMERGENCY_SAVE_{real_timesteps}.zip",
                        f"Otimizacao/treino_principal/EMERGENCY_{real_timesteps}.zip"
                    ]
                    
                    for i, emergency_path in enumerate(emergency_paths):
                        try:
                            print(f"🆘 Tentativa emergência #{i+1}: {emergency_path}")
                            if os.path.dirname(emergency_path):  # Só se não for string vazia
                                os.makedirs(os.path.dirname(emergency_path), exist_ok=True)
                            
                            # 🔥 SALVAMENTO SIMPLES SEM CLOUDPICKLE
                            self.model.save(emergency_path)
                            
                            if os.path.exists(emergency_path):
                                size_mb = os.path.getsize(emergency_path) / (1024*1024)
                                print(f"🆘 Emergência #{i+1}: {size_mb:.1f}MB")
                                
                                # Verificar conteúdo
                                try:
                                    import zipfile
                                    with zipfile.ZipFile(emergency_path, 'r') as z:
                                        files = [f.filename for f in z.filelist]
                                        if any('policy.pth' in f for f in files):
                                            print(f"🆘 Emergência #{i+1} COMPLETA!")
                                            successful_saves += 1
                                            break
                                        else:
                                            print(f"🆘 Emergência #{i+1} incompleta: {files}")
                                except:
                                    if size_mb > 2.5:
                                        successful_saves += 1
                                        break
                        except Exception as emergency_error:
                            print(f"🆘 Emergência #{i+1} falhou: {emergency_error}")
                            
                elif successful_saves < 3:
                    print(f"⚠️ Apenas {successful_saves}/3 salvamentos - criando backup adicional")
                    try:
                        backup_path = f"trading_framework/training/BACKUP_EXTRA_{real_timesteps}_{timestamp}.zip"
                        self.model.save(backup_path)
                        print(f"💾 Backup adicional criado: {backup_path}")
                    except:
                        print("❌ Backup adicional falhou")
                
                print(f">>> 💾 CHECKPOINT STEP {real_timesteps:,} COMPLETO <<<\n")
                
            except Exception as e:
                print(f"❌ ERRO CRÍTICO no salvamento step {real_timesteps:,}: {e}")
                import traceback
                traceback.print_exc()
                
                # 🆘 ÚLTIMO RECURSO - SALVAMENTO SIMPLES
                try:
                    emergency_path = f"LAST_RESORT_{real_timesteps}.zip"
                    print(f"🆘 ÚLTIMO RECURSO: {emergency_path}")
                    self.model.save(emergency_path)
                    print("🆘 Último recurso executado")
                except Exception as final_error:
                    print(f"🆘 ÚLTIMO RECURSO FALHOU: {final_error}")
                    print("🚨 SISTEMA DE SALVAMENTO COMPLETAMENTE FALHOU!")
                    
        return True

def _determine_phase_from_steps(steps):
    """Determinar índice da fase baseado nos steps (dobro do número de barras)"""
    # 🎯 CALCULAR TOTAL BASEADO NO DATASET FINAL (723,547 barras * 2 = 1,447,094 steps)
    total_steps = 1447094  # Dobro do dataset final sem preços estáticos
    
    # Fases: 20%, 25%, 25%, 20%, 10% do total
    phase_thresholds = [
        int(total_steps * 0.20),   # Fase 1: 0 - 289k (20% do total)
        int(total_steps * 0.45),   # Fase 2: 289k - 651k (25% adicional)
        int(total_steps * 0.70),   # Fase 3: 651k - 1.013M (25% adicional)
        int(total_steps * 0.90),   # Fase 4: 1.013M - 1.302M (20% adicional)
        total_steps                # Fase 5: 1.302M - 1.447M (10% adicional)
    ]
    
    for i, threshold in enumerate(phase_thresholds):
        if steps < threshold:
            return i
    
    # Se passou de todas as fases, está na última
    return len(phase_thresholds) - 1

def _create_training_phases() -> List[TrainingPhase]:
    """🔥 FASES AJUSTADAS PARA O DATASET FINAL SEM PREÇOS ESTÁTICOS (1.447M steps total)"""
    # 🎯 CALCULAR TOTAL BASEADO NO DATASET FINAL (723,547 barras * 2 = 1,447,094 steps)
    total_steps = 1447094  # Dobro do dataset final sem preços estáticos
    
    return [
        TrainingPhase(
            name="Phase_1_Fundamentals",
            phase_type=PhaseType.FUNDAMENTALS,
            timesteps=int(total_steps * 0.20),  # 289k steps (20% do total)
            description="Aprender reconhecimento básico de tendências",
            data_filter="trending",
            success_criteria={
                "win_rate": 0.99,  # Critério impossível para evitar early stop
                "trades_per_hour": 999
            },
            reset_criteria={
                "win_rate": 0.25,
                "max_drawdown": 0.30
            }
        ),
        TrainingPhase(
            name="Phase_2_Risk_Management", 
            phase_type=PhaseType.RISK_MANAGEMENT,
            timesteps=int(total_steps * 0.25),  # 362k steps (25% do total)
            description="Dominar uso de SL/TP e gestão de risco",
            data_filter="reversal_periods",
            success_criteria={
                "max_drawdown": -999,  # Impossível para evitar early stop
                "win_rate": 0.99
            },
            reset_criteria={
                "max_drawdown": 0.35,
                "win_rate": 0.30
            }
        ),
        TrainingPhase(
            name="Phase_3_Noise_Handling",
            phase_type=PhaseType.NOISE_HANDLING, 
            timesteps=int(total_steps * 0.25),  # 362k steps (25% do total)
            description="Evitar overtrading em mercados laterais",
            data_filter="sideways",
            success_criteria={
                "sharpe_ratio": 999,  # Impossível para evitar early stop
                "win_rate": 0.99
            },
            reset_criteria={
                "sharpe_ratio": -0.2,
                "win_rate": 0.35
            }
        ),
        TrainingPhase(
            name="Phase_4_Stress_Testing",
            phase_type=PhaseType.STRESS_TESTING,
            timesteps=int(total_steps * 0.20),  # 289k steps (20% do total)
            description="Lidar com volatilidade extrema e eventos de cauda",
            data_filter="high_volatility",
            success_criteria={
                "tail_risk_ratio": 999,  # Impossível para evitar early stop
                "volatility_adjusted_return": 999
            },
            reset_criteria={
                "max_drawdown": 0.25,
                "tail_risk_ratio": 0.7
            }
        ),
        TrainingPhase(
            name="Phase_5_Integration",
            phase_type=PhaseType.INTEGRATION,
            timesteps=int(total_steps * 0.10),  # 145k steps (10% do total)
            description="Integrar todas as habilidades em dataset completo",
            data_filter="mixed",
            success_criteria={
                "sharpe_ratio": 999,  # Impossível para evitar early stop
                "max_drawdown": -999,
                "win_rate": 0.99
            },
            reset_criteria={
                "sharpe_ratio": 0.5,
                "max_drawdown": 0.15
            }
        )
    ]

# === FUNÇÃO PRINCIPAL ATUALIZADA ===

# 🔥 SISTEMA DE RESUME TRAINING
def find_latest_checkpoint():
    """Encontrar o último checkpoint salvo"""
    checkpoint_dir = "Otimizacao/treino_principal/models/DIFF"
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # Procurar por arquivos de modelo
    model_files = glob.glob(os.path.join(checkpoint_dir, "DIFF_model_*.zip"))
    if not model_files:
        return None, 0
    
    # Encontrar o mais recente
    latest_file = max(model_files, key=os.path.getctime)
    
    # Extrair steps do nome do arquivo
    filename = os.path.basename(latest_file)
    try:
        # Formato esperado: DIFF_model_XXXXXX.zip
        steps_str = filename.replace("DIFF_model_", "").replace(".zip", "")
        steps = int(steps_str)
        return latest_file, steps
    except:
        return latest_file, 0

def main():
    """Função principal do treinamento DIFF com estrutura de fases do mainppo1.py"""
    print("🚀 INICIANDO TREINAMENTO DIFERENCIADO PPO COM FASES")
    print("🎯 Configuração: 1.447M steps total em 5 fases (dobro do dataset final)")
    print("📊 Target: 25 trades/dia (🔥 AUMENTADO para exploração agressiva)")
    print("💰 SL/TP: 8-60 SL, 12-80 TP")
    print("🎯 Dataset: GOLD_final_nostatic.pkl (723,547 barras sem preços estáticos)")
    
    # Verificar se há checkpoint para continuar
    latest_checkpoint, steps_completed = find_latest_checkpoint()
    if latest_checkpoint and steps_completed > 0:
        print(f"🔄 RESUMINDO TREINAMENTO: {steps_completed:,} steps já completados")
        print(f"📁 Checkpoint: {os.path.basename(latest_checkpoint)}")
    else:
        print("🚀 INICIANDO TREINAMENTO DO ZERO")
    
    # Carregar dados
    df = load_optimized_data()
    print(f"✅ Dataset carregado: {len(df):,} barras")
    
    # 🎯 CALCULAR TOTAL DE STEPS BASEADO NO DATASET
    total_steps = len(df) * 2  # Dobro do número de barras
    print(f"🎯 Total de steps planejado: {total_steps:,} (dobro de {len(df):,} barras)")
    
    # Criar ambiente
    # 🚀 USAR AMBIENTE FRAMEWORK COM POSIÇÕES REAIS
    from trading_framework.environments.trading_env import TradingEnv as FrameworkTradingEnv
    
    env = FrameworkTradingEnv(
        df=df,
        window_size=20, 
        is_training=True,
        initial_balance=1000,
        reward_system_type="diff_reward"  # Sistema diferenciado
    )
    print("✅ Ambiente criado")
    
    # Adicionar VecNormalize para normalização de observações e recompensas
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
    # Envolver ambiente com VecNormalize
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    print("✅ VecNormalize configurado (obs + rewards)")
    
    # Importar TwoHeadPolicy e TradingTransformerFeatureExtractor
    from trading_framework.policies.two_head_policy import TwoHeadPolicy
    from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
    
    # Configurar TradingTransformerFeatureExtractor
    feature_extractor_kwargs = {
        "features_dim": 128,
        "seq_len": 8
    }
    
    # Criar modelo com TwoHeadPolicy + TradingTransformerFeatureExtractor
    if latest_checkpoint and steps_completed > 0:
        # 🔥 CARREGAR MODELO EXISTENTE
        print(f"🔄 Carregando modelo existente: {os.path.basename(latest_checkpoint)}")
        model = RecurrentPPO.load(latest_checkpoint, env=env)
        print(f"✅ Modelo carregado com {steps_completed:,} steps")
        
        # Carregar VecNormalize se existir
        vecnorm_path = latest_checkpoint.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(vecnorm_path):
            try:
                env = VecNormalize.load(vecnorm_path, env)
                print("✅ VecNormalize carregado")
            except Exception as e:
                print(f"⚠️ Erro ao carregar VecNormalize: {e}")
    else:
        # 🔥 CRIAR NOVO MODELO
        model = RecurrentPPO(
            TwoHeadPolicy,
            env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,  # 🔥 AUMENTADO: 0.01 → 0.1 para mais exploração
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./logs/",
            policy_kwargs={
                "features_extractor_class": TradingTransformerFeatureExtractor,
                "features_extractor_kwargs": feature_extractor_kwargs,
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1
            }
        )
        print("✅ Novo modelo PPO criado")
    
    # 🎯 SISTEMA DE FASES
    phases = _create_training_phases()
    metrics_tracker = PhaseMetrics()
    adaptive_reset = AdaptiveReset()
    
    print(f"\n📋 ESTRUTURA DE FASES:")
    total_timesteps = sum(p.timesteps for p in phases)
    print(f"   Total de timesteps: {total_timesteps:,}")
    for i, phase in enumerate(phases):
        print(f"   Fase {i+1}: {phase.name} - {phase.timesteps:,} steps - {phase.description}")
    
    # Callbacks para métricas
    from stable_baselines3.common.callbacks import BaseCallback
    import os
    from collections import deque
    
    class AdvancedMetricsCallback(BaseCallback):
        def __init__(self, verbose=0, steps_offset=0):  # 🔥 ADICIONADO: steps_offset
            super().__init__(verbose)
            self.steps_offset = steps_offset  # 🔥 ADICIONADO: Offset de steps para resume
            
            # 🎯 HISTÓRICO GLOBAL DE TRADES - NÃO RESETADO A CADA EPISÓDIO
            self.global_trades_history = []  # Lista global de todos os trades
            self.global_total_trades = 0  # 🔥 CORREÇÃO: Contador global de trades
            self.global_total_pnl = 0.0
            self.global_winning_trades = 0  # 🔥 CORREÇÃO: Adicionar contador de trades vencedores
            self.global_portfolio_peak = 1000.0
            self.global_start_time = 0  # 🔥 CORREÇÃO: Inicializar com 0, não None
            
            # 🎯 MÉTRICAS ROLANTES
            self.recent_portfolios = deque(maxlen=20)
            self.recent_trades_counts = deque(maxlen=20)
            self.recent_win_rates = deque(maxlen=20)
            self.recent_pnls = deque(maxlen=20)
            self.recent_drawdowns = deque(maxlen=20)
            
            print(f"🔧 AdvancedMetricsCallback inicializado com histórico global de trades (offset: {steps_offset:,} steps)")
        
        def _on_step(self) -> bool:
            # Coletar métricas do ambiente
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                
                # 🔥 CORREÇÃO: Atualizar global_step_offset do ambiente com step acumulado real
                current_total_steps = self.num_timesteps + self.steps_offset
                env.global_step_offset = current_total_steps - env.current_step
                
                # 🔥 DEBUG: Verificar cálculo do global_step_offset
                if self.num_timesteps % 10000 == 0:  # Debug a cada 10k steps
                    print(f"🔍 DEBUG GLOBAL_STEP_OFFSET:")
                    print(f"   num_timesteps: {self.num_timesteps:,}")
                    print(f"   steps_offset: {self.steps_offset:,}")
                    print(f"   current_total_steps: {current_total_steps:,}")
                    print(f"   env.current_step: {env.current_step:,}")
                    print(f"   global_step_offset: {env.global_step_offset:,}")
                
                if hasattr(env, 'trades'):
                    trades_count = len(env.trades)
                    # 🔥 CORREÇÃO: Usar portfolio global acumulado
                    portfolio_value = env.portfolio_value
                    
                    # 🔥 CORREÇÃO: Atualizar contador global de trades
                    if trades_count > self.global_total_trades:
                        # Novos trades foram adicionados
                        new_trades = trades_count - self.global_total_trades
                        self.global_total_trades = trades_count
                        
                        # Atualizar histórico global
                        for i in range(new_trades):
                            if i < len(env.trades):
                                trade = env.trades[-(i+1)]  # Pegar trades mais recentes
                                self.global_trades_history.append(trade)
                    
                    # Atualizar peak do portfolio
                    if portfolio_value > self.global_portfolio_peak:
                        self.global_portfolio_peak = portfolio_value
                    
                    # Calcular métricas básicas
                    if trades_count > 0:
                        winning_trades = [t for t in env.trades if t.get('pnl_usd', 0) > 0]
                        win_rate = len(winning_trades) / trades_count
                        total_pnl = sum(t.get('pnl_usd', 0) for t in env.trades)
                        avg_pnl = total_pnl / trades_count
                        
                        self.recent_win_rates.append(win_rate)
                        self.recent_pnls.append(avg_pnl)
                    
                    self.recent_portfolios.append(portfolio_value)
                    self.recent_trades_counts.append(trades_count)
                    
                    # MÉTRICAS A CADA 1000 STEPS - SIMPLES E FUNCIONAL
                    if self.num_timesteps % 1000 == 0:
                        # 🔥 CORREÇÃO: Usar steps totais (incluindo offset)
                        total_steps = self.num_timesteps + self.steps_offset
                        
                        # Calcular métricas básicas
                        avg_portfolio = np.mean(list(self.recent_portfolios)[-20:]) if self.recent_portfolios else 1000
                        avg_win_rate = np.mean(list(self.recent_win_rates)[-20:]) if self.recent_win_rates else 0
                        avg_pnl = np.mean(list(self.recent_pnls)[-20:]) if self.recent_pnls else 0
                        
                        # 🔥 CORREÇÃO: Calcular trades/dia CORRETO
                        steps_elapsed = total_steps
                        days_elapsed = steps_elapsed / 288  # 288 barras = 1 dia
                        
                        # 🔥 CORREÇÃO: Trades/dia total (todos os trades / todos os dias)
                        total_trades_accumulated = len(self.global_trades_history)
                        trades_per_day = total_trades_accumulated / days_elapsed if days_elapsed > 0 else 0
                        
                        # 🎯 TRADES/DIA GLOBAL - ÚLTIMOS 100 DIAS EXATOS - CORRIGIDO
                        days_100_steps = 100 * 288  # Exatamente 100 dias = 28,800 steps
                        threshold_100_days = total_steps - days_100_steps  # Últimos 100 dias EXATOS
                        
                        # 🚨 CORREÇÃO: Mas ainda filtrar trades de outras sessões
                        min_valid_step = total_steps - 10000  # Últimos 10k steps (sessão atual)
                        current_session_threshold = max(min_valid_step, threshold_100_days)
                        
                        recent_trades = 0
                        
                        for trade in reversed(self.global_trades_history):
                            trade_step = trade.get('step', 0)
                            if trade_step >= current_session_threshold:
                                recent_trades += 1
                            else:
                                break
                        
                        # 🔥 DEBUG: Verificar cálculo dos últimos 100 dias - CORRIGIDO
                        if total_steps % 10000 == 0:  # Debug a cada 10k steps
                            print(f"🔍 DEBUG ÚLTIMOS 100 DIAS - CORRIGIDO:")
                            print(f"   Total steps: {total_steps:,}")
                            print(f"   Steps offset: {self.steps_offset:,}")
                            print(f"   Days 100 steps: {days_100_steps:,} (100 dias exatos)")
                            print(f"   ❌ OLD threshold: {total_steps - days_100_steps:,}")
                            print(f"   🔍 Min valid step: {min_valid_step:,}")  
                            print(f"   ✅ NEW threshold: {current_session_threshold:,}")
                            print(f"   Total trades no histórico: {len(self.global_trades_history)}")
                            print(f"   ✅ Trades encontrados (CORRIGIDO): {recent_trades}")
                            
                            # 🔥 DEBUG DETALHADO: Verificar steps dos trades
                            if len(self.global_trades_history) > 0:
                                latest_trade_step = self.global_trades_history[-1].get('step', 0)
                                oldest_trade_step = self.global_trades_history[0].get('step', 0)
                                print(f"   Último trade step: {latest_trade_step:,}")
                                print(f"   Primeiro trade step: {oldest_trade_step:,}")
                                print(f"   Range dos trades: {oldest_trade_step:,} - {latest_trade_step:,}")
                                
                                # 🔥 VERIFICAR SE TRADES ESTÃO COM STEP CORRETO
                                sample_trades = self.global_trades_history[-5:]  # Últimos 5 trades
                                print(f"   Últimos 5 trades steps:")
                                for i, trade in enumerate(sample_trades):
                                    trade_step = trade.get('step', 0)
                                    print(f"     Trade {i+1}: {trade_step:,}")
                                
                                # 🔥 VERIFICAR SE ALGUM TRADE PASSA NO FILTRO - CORRIGIDO
                                old_threshold = total_steps - days_100_steps
                                new_threshold = current_session_threshold
                                trades_above_old = [t for t in self.global_trades_history if t.get('step', 0) >= old_threshold]
                                trades_above_new = [t for t in self.global_trades_history if t.get('step', 0) >= new_threshold]
                                
                                print(f"   ❌ Trades acima OLD threshold ({old_threshold:,}): {len(trades_above_old)}")
                                print(f"   ✅ Trades acima NEW threshold ({new_threshold:,}): {len(trades_above_new)}")
                                
                                if len(trades_above_new) > 0:
                                    print(f"   ✅ CORREÇÃO FUNCIONOU! Exemplo: step {trades_above_new[0].get('step', 0):,}")
                                else:
                                    print(f"   ⚠️ Ainda sem trades no período atual (sessão começou recentemente)")
                            else:
                                print(f"   ❌ HISTÓRICO DE TRADES VAZIO!")
                                print(f"   ❌ PROBLEMA: global_trades_history não está sendo preenchido")
                        
                        # 🔥 CORREÇÃO: Trades/dia dos EXATOS 100 DIAS
                        global_trades_per_day = recent_trades / 100  # Sempre dividir por 100 dias exatos
                        
                        # 🎯 DRAWDOWN MÉDIO - ÚLTIMOS 100 DIAS
                        recent_drawdowns = []
                        portfolios_list = list(self.recent_portfolios)  # Converter deque para lista
                        for i in range(max(0, len(portfolios_list) - 100), len(portfolios_list)):
                            if i > 0:
                                peak = max(portfolios_list[:i+1])
                                current = portfolios_list[i]
                                dd = (peak - current) / peak if peak > 0 else 0
                                recent_drawdowns.append(dd)
                        avg_drawdown = np.mean(recent_drawdowns) if recent_drawdowns else 0
                        
                        # 🎯 DURAÇÃO MÉDIA - TODOS OS TRADES RECENTES (máximo 100)
                        recent_durations = []
                        # 🔥 CORREÇÃO: Usar apenas trades que passaram no filtro de período
                        for trade in reversed(self.global_trades_history):
                            trade_step = trade.get('step', 0)
                            if trade_step >= current_session_threshold:  # Mesmo filtro usado acima
                                if len(recent_durations) >= 100:  # Máximo 100 trades
                                    break
                                duration = trade.get('duration', 0)
                                if duration > 0:
                                    recent_durations.append(duration)
                        
                        # 🔥 MOSTRAR QUANTOS TRADES REALMENTE TEMOS
                        trades_count_for_duration = len(recent_durations)
                        avg_duration_hours = np.mean(recent_durations) * 5 / 60 if recent_durations else 0
                        
                        # Determinar fase
                        current_phase_idx = _determine_phase_from_steps(total_steps)
                        current_phase = phases[current_phase_idx] if current_phase_idx < len(phases) else None
                        
                        # LOG SIMPLES E FUNCIONAL
                        print(f"\n📊 MÉTRICAS - Step {total_steps:,}")
                        print(f"   🎯 Fase: {current_phase_idx + 1}/5 - {current_phase.name if current_phase else 'Final'}")
                        print(f"   💰 Portfolio: ${portfolio_value:.2f}")
                        print(f"   📈 Portfolio Peak: ${self.global_portfolio_peak:.2f}")
                        print(f"   🎯 Trades/dia: {trades_per_day:.1f}")
                        print(f"   🎯 Trades/dia (últimos 100 dias): {global_trades_per_day:.1f}")
                        print(f"   🏆 Win rate: {avg_win_rate:.1%}")
                        print(f"   📊 PnL médio: ${avg_pnl:.2f}")
                        print(f"   📉 DD médio (últimos 100 dias): {avg_drawdown:.1%}")
                        print(f"   ⏱️ Duração média ({trades_count_for_duration} trades): {avg_duration_hours:.1f}h")
                        print(f"   📊 Total trades: {total_trades_accumulated}")
                        print(f"   📅 Dias decorridos: {days_elapsed:.1f}")

            return True
        
        def _on_rollout_end(self) -> None:
            """Chamado no final de cada rollout"""
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                if hasattr(env, 'trades'):
                    trades_count = len(env.trades)
                    # 🔥 CORREÇÃO: Usar portfolio global acumulado
                    portfolio_value = env.portfolio_value
                    
                    # 🔥 CORREÇÃO: Calcular métricas do rollout CORRETO
                    steps_elapsed = self.num_timesteps - self.global_start_time
                    days_elapsed = steps_elapsed / 288
                    # 🔥 CORREÇÃO: Usar fórmula correta para trades/dia no rollout
                    trades_per_day_rollout = trades_count / days_elapsed if days_elapsed > 0 else 0
                    
                    print(f"\n🔄 ROLLOUT COMPLETO - {trades_count} trades, Portfolio: ${portfolio_value:.2f}")
                    print(f"   📊 Trades/dia neste rollout: {trades_per_day_rollout:.1f}")
                    
                    # Resetar contadores para próximo episódio
                    if hasattr(env, 'reset'):
                        env.reset()
    
    # Criar callbacks
    metrics_callback = AdvancedMetricsCallback(steps_offset=steps_completed)
    save_callback = RobustSaveCallback(
        save_freq=10000,
        save_path="Otimizacao/treino_principal/models/DIFF",
        name_prefix="DIFF_model",
        total_steps_offset=0,  # 🔥 NOVO: Offset para steps acumulados
        training_env=env  # 🔥 CORREÇÃO: Passar environment para callback
    )
    
    # Treinar modelo com estrutura de fases
    print("🎯 Iniciando treinamento com estrutura de fases...")
    
    # 🔥 CALCULAR STEPS RESTANTES
    total_timesteps = sum(p.timesteps for p in phases)
    remaining_steps = total_timesteps - steps_completed
    
    if remaining_steps > 0:
        print(f"📊 Steps restantes: {remaining_steps:,}")
        
        # Treinar modelo com callbacks
        model.learn(
            total_timesteps=remaining_steps, 
            progress_bar=True,
            callback=[metrics_callback, save_callback],
            reset_num_timesteps=False  # 🔥 IMPORTANTE: Não resetar contador de steps
        )
        
        print("✅ Treinamento concluído!")
    else:
        print("✅ Treinamento já concluído!")
    
    # 💾 SALVAR MODELO FINAL E VECNORMALIZE
    print("💾 Salvando modelo final e VecNormalize...")
    
    # Salvar modelo final
    model.save("Otimizacao/treino_principal/models/DIFF/DIFF_model_final")
    print("✅ Modelo final salvo!")
    
    # Salvar VecNormalize final
    try:
        vecnorm_final_path = "Otimizacao/treino_principal/models/DIFF/vec_normalize_final.pkl"
        env.save(vecnorm_final_path)
        print("✅ VecNormalize final salvo!")
        print(f"   📁 Arquivo: {vecnorm_final_path}")
    except Exception as e:
        print(f"❌ Erro ao salvar VecNormalize: {e}")
        # Tentar método alternativo
        try:
            if hasattr(env, 'save_running_average'):
                env.save_running_average("Otimizacao/treino_principal/models/DIFF/vec_normalize_final.pkl")
                print("✅ VecNormalize salvo via método alternativo!")
            else:
                print("⚠️ VecNormalize não pôde ser salvo - método não disponível")
        except Exception as e2:
            print(f"❌ Erro no método alternativo: {e2}")
    
    print("🎉 Treinamento DIFF concluído com sucesso!")

if __name__ == "__main__":
    main() 