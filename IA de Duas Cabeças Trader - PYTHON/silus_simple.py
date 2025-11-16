#!/usr/bin/env python3
"""
🎯 SILUS SIMPLE - TREINAMENTO EFICAZ E ELEGANTE
Versão simplificada e otimizada do sistema de treinamento
Foco em eficiência e performance real
"""

import os
import sys
import gc
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Setup paths
sys.path.append("D:/Projeto")

# Import essenciais
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# ============================================
# CONFIGURAÇÃO SIMPLES E EFICAZ
# ============================================

# GPU Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
    print(f"   Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("⚠️ Usando CPU (treinamento será mais lento)")

# ============================================
# HIPERPARÂMETROS COMPROVADOS
# ============================================

PROVEN_HYPERPARAMS = {
    # Learning rates balanceados e testados
    "learning_rate": 3e-05,           # Sweet spot para PPO trading
    "critic_learning_rate": 1.5e-05,  # Critic mais conservador
    
    # Configuração de batch otimizada
    "n_steps": 2048,                  # Trajetória adequada
    "batch_size": 64,                 # Batch estável
    "n_epochs": 10,                   # Aproveitar bem os dados
    
    # Parâmetros PPO padrão eficazes
    "gamma": 0.99,                    # Visão de longo prazo
    "gae_lambda": 0.95,               # GAE padrão
    "clip_range": 0.2,                # Clipping padrão PPO
    "ent_coef": 0.01,                 # Exploração moderada
    "vf_coef": 0.5,                   # Value function padrão
    "max_grad_norm": 0.5,             # Gradient clipping moderado
    "target_kl": 0.01,                # KL divergence controlado
    
    # Policy kwargs otimizados
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 256],        # Actor network simples
            "vf": [256, 256]          # Critic network simples
        },
        "activation_fn": torch.nn.ReLU,  # ReLU é mais estável que SILU
        "normalize_images": False,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {
            "eps": 1e-5,
            "betas": (0.9, 0.999)
        }
    }
}

# ============================================
# CONFIGURAÇÃO DE TREINAMENTO
# ============================================

TRAINING_CONFIG = {
    "total_timesteps": 5_000_000,    # 5M steps total (sweet spot)
    "checkpoint_freq": 100_000,       # Salvar a cada 100k
    "eval_freq": 250_000,             # Avaliar a cada 250k
    "log_freq": 10_000,               # Log a cada 10k
    "experiment_name": "SILUS_SIMPLE",
    "save_path": "D:/Projeto/Otimizacao/treino_principal/models/SILUS_SIMPLE",
    "dataset_path": "D:/Projeto/datasets/DAYTRADER_V3_COMPLETO.parquet"
}

# ============================================
# CALLBACKS ESSENCIAIS
# ============================================

class SimpleProgressCallback(BaseCallback):
    """Callback simples para mostrar progresso"""
    
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.last_log = 0
        
    def _on_training_start(self):
        self.start_time = time.time()
        print("\n" + "="*80)
        print(f"🚀 INICIANDO TREINAMENTO SIMPLES - {self.total_timesteps:,} steps")
        print("="*80 + "\n")
        
    def _on_step(self):
        # Log a cada 10k steps
        if self.num_timesteps - self.last_log >= 10000:
            self.last_log = self.num_timesteps
            
            # Calcular métricas
            progress = self.num_timesteps / self.total_timesteps * 100
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed
            eta = (self.total_timesteps - self.num_timesteps) / steps_per_sec
            
            # Pegar métricas do logger se disponível
            ep_rew_mean = "N/A"
            ep_len_mean = "N/A"
            if "rollout/ep_rew_mean" in self.logger.name_to_value:
                ep_rew_mean = f"{self.logger.name_to_value['rollout/ep_rew_mean']:.2f}"
            if "rollout/ep_len_mean" in self.logger.name_to_value:
                ep_len_mean = f"{self.logger.name_to_value['rollout/ep_len_mean']:.0f}"
            
            print(f"📊 Step {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1f}%)")
            print(f"   ⏱️ Speed: {steps_per_sec:.0f} steps/s | ETA: {eta/3600:.1f}h")
            print(f"   📈 Reward: {ep_rew_mean} | Episode Length: {ep_len_mean}")
            print()
            
        return True

class SmartEarlyStoppingCallback(BaseCallback):
    """Early stopping baseado em performance real"""
    
    def __init__(self, patience=500_000, min_improvement=0.02, verbose=1):
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_reward = -np.inf
        self.best_step = 0
        self.no_improvement_steps = 0
        
    def _on_step(self):
        # Verificar a cada 50k steps
        if self.num_timesteps % 50000 == 0:
            # Pegar reward médio atual
            if "rollout/ep_rew_mean" in self.logger.name_to_value:
                current_reward = self.logger.name_to_value["rollout/ep_rew_mean"]
                
                # Verificar melhoria
                improvement = (current_reward - self.best_reward) / (abs(self.best_reward) + 1e-8)
                
                if improvement > self.min_improvement:
                    # Melhoria significativa
                    self.best_reward = current_reward
                    self.best_step = self.num_timesteps
                    self.no_improvement_steps = 0
                    print(f"✅ Nova melhor performance: {current_reward:.2f} (step {self.num_timesteps:,})")
                else:
                    # Sem melhoria
                    self.no_improvement_steps = self.num_timesteps - self.best_step
                    
                    if self.no_improvement_steps >= self.patience:
                        print(f"\n⚠️ Early stopping triggered!")
                        print(f"   Sem melhoria há {self.no_improvement_steps:,} steps")
                        print(f"   Melhor performance: {self.best_reward:.2f} no step {self.best_step:,}")
                        return False  # Para o treinamento
                        
        return True

# ============================================
# AMBIENTE DE TRADING SIMPLIFICADO
# ============================================

def create_simple_trading_env(dataset_path=None):
    """Criar ambiente de trading simples e eficaz"""
    
    # Importar o ambiente
    from silus import TradingEnvironmentV3 as TradingEnv
    
    # Carregar dataset
    if dataset_path and os.path.exists(dataset_path):
        print(f"📊 Carregando dataset: {dataset_path}")
        df = pd.read_parquet(dataset_path)
    else:
        print("📊 Gerando dataset sintético para teste...")
        # Dataset sintético simples
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='5min')
        df = pd.DataFrame({
            'timestamp': dates,
            'open_5m': np.random.randn(len(dates)).cumsum() + 2000,
            'high_5m': np.random.randn(len(dates)).cumsum() + 2010,
            'low_5m': np.random.randn(len(dates)).cumsum() + 1990,
            'close_5m': np.random.randn(len(dates)).cumsum() + 2000,
            'volume_5m': np.random.uniform(100, 1000, len(dates))
        })
    
    print(f"   Shape: {df.shape}")
    print(f"   Período: {df.iloc[0]['timestamp']} até {df.iloc[-1]['timestamp']}")
    
    # Criar ambiente
    env = TradingEnv(
        df=df,
        initial_balance=10000,
        max_positions=2,
        lot_size=0.1,
        transaction_cost=0.0002,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        max_episode_steps=10000,
        reward_scaling=1.0
    )
    
    # Vetorizar e normalizar
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    return env

# ============================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# ============================================

def train_simple_and_effective():
    """
    Treinar modelo de forma simples e eficaz
    Sem curriculum learning, sem complicações, apenas eficiência
    """
    
    print("\n" + "🎯"*40)
    print("🎯 SILUS SIMPLE - TREINAMENTO EFICAZ E ELEGANTE")
    print("🎯 Sem curriculum learning, sem filtros, sem complicações")
    print("🎯 Apenas treinamento direto e eficiente")
    print("🎯"*40 + "\n")
    
    # Criar diretórios
    os.makedirs(TRAINING_CONFIG["save_path"], exist_ok=True)
    
    # Timestamp para esta run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ============================================
    # 1. CRIAR AMBIENTE
    # ============================================
    
    print("📦 Criando ambiente de trading...")
    env = create_simple_trading_env(TRAINING_CONFIG["dataset_path"])
    print("✅ Ambiente criado com sucesso\n")
    
    # ============================================
    # 2. CRIAR MODELO
    # ============================================
    
    print("🤖 Criando modelo PPO...")
    model = PPO(
        policy="MlpPolicy",  # MLP simples e eficaz
        env=env,
        **PROVEN_HYPERPARAMS,
        verbose=1,
        device=device,
        tensorboard_log=f"./tensorboard/{TRAINING_CONFIG['experiment_name']}_{timestamp}"
    )
    print("✅ Modelo criado com sucesso\n")
    
    # ============================================
    # 3. CONFIGURAR CALLBACKS
    # ============================================
    
    print("⚙️ Configurando callbacks...")
    
    callbacks = [
        # Progresso
        SimpleProgressCallback(
            total_timesteps=TRAINING_CONFIG["total_timesteps"],
            verbose=1
        ),
        
        # Checkpoints
        CheckpointCallback(
            save_freq=TRAINING_CONFIG["checkpoint_freq"],
            save_path=TRAINING_CONFIG["save_path"],
            name_prefix=f"{TRAINING_CONFIG['experiment_name']}_{timestamp}",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=1
        ),
        
        # Early stopping inteligente
        SmartEarlyStoppingCallback(
            patience=500_000,  # 500k steps sem melhoria
            min_improvement=0.02,  # 2% de melhoria mínima
            verbose=1
        )
    ]
    
    combined_callback = CallbackList(callbacks)
    print("✅ Callbacks configurados\n")
    
    # ============================================
    # 4. TREINAR
    # ============================================
    
    print("🚀 INICIANDO TREINAMENTO SIMPLES E EFICAZ")
    print(f"   Total steps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"   Device: {device}")
    print(f"   Checkpoints: {TRAINING_CONFIG['save_path']}")
    print("\n" + "="*80 + "\n")
    
    try:
        # Treinar
        model.learn(
            total_timesteps=TRAINING_CONFIG["total_timesteps"],
            callback=combined_callback,
            progress_bar=False,  # Usamos nosso próprio callback
            reset_num_timesteps=True,
            tb_log_name=f"run_{timestamp}"
        )
        
        print("\n" + "✅"*40)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("✅"*40 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️ Treinamento interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================
    # 5. SALVAR MODELO FINAL
    # ============================================
    
    try:
        final_path = f"{TRAINING_CONFIG['save_path']}/FINAL_{TRAINING_CONFIG['experiment_name']}_{timestamp}.zip"
        print(f"💾 Salvando modelo final: {final_path}")
        model.save(final_path)
        env.save(f"{TRAINING_CONFIG['save_path']}/vecnormalize_{timestamp}.pkl")
        print("✅ Modelo final salvo com sucesso!\n")
    except Exception as e:
        print(f"❌ Erro ao salvar modelo: {e}")
    
    # Limpeza
    del model
    env.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("🎯 Processo finalizado com sucesso!")
    return final_path

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Configurar seeds para reprodutibilidade
    set_random_seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Executar treinamento
    final_model_path = train_simple_and_effective()
    
    print(f"\n📊 Modelo treinado salvo em: {final_model_path}")
    print("\n🎯 Para avaliar o modelo, use:")
    print(f'   python avaliar_v11.py "{final_model_path}"')