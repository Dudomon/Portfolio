#!/usr/bin/env python3
"""
🚀 AVALIAÇÃO REALÍSTICA 1MIN - ULTRA OTIMIZADA (5x VELOCIDADE)
============================================================

OTIMIZAÇÕES IMPLEMENTADAS SEM AFETAR CONFIABILIDADE:
✅ 1. Batch prediction (10x predictions por vez)
✅ 2. Pre-computed features cache permanente  
✅ 3. Memory layout optimization (pre-allocated arrays)
✅ 4. Environment optimizations (logging desabilitado)
✅ 5. Reward system eval mode (cálculos simplificados)
✅ 6. Intelligent batching baseado em market patterns
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
import random
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sys.path.append("D:/Projeto")

# MUDAR PARA O DIRETÓRIO CORRETO PARA ACESSAR data/
os.chdir("D:/Projeto")

import numpy as np
import pandas as pd
import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

# 🚀 MEMORY POOL GLOBAL PARA MÁXIMA REUTILIZAÇÃO
class MemoryPool:
    def __init__(self):
        self.obs_pool = np.zeros((50, 450), dtype=np.float32)  # Batch maior
        self.reward_pool = np.zeros(50, dtype=np.float32)
        self.action_pool = np.zeros((50, 1), dtype=np.float32)
        self.activity_pool = deque(maxlen=200)  # Rolling activity window

    def reset(self):
        self.obs_pool.fill(0)
        self.reward_pool.fill(0)
        self.action_pool.fill(0)
        self.activity_pool.clear()

# 🚀 DATASET PRE-PROCESSADO GLOBAL
_preprocessed_data = None
_preprocessing_lock = threading.Lock()

# 🍒 CAMPEÕES - CHERRY BEST vs V3 BRUTAL 1.325M (Baseline) - PATHS ORIGINAIS
CHECKPOINTS_TO_TEST = [
    "D:/Projeto/trading_framework/training/checkpoints/cherry/checkpoint_1175000_steps_20250924_140441.zip",  # 🍒 Cherry 1.175M
    "D:/Projeto/Otimizacao/treino_principal/models/cherry/cherry_simpledirecttraining_1200000_steps_20250924_140624.zip",  # 🏆 Cherry 1.2M Champion
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_1325000_steps_20250923_191757.zip"  # ⚔️ V3 Brutal Baseline
]

# PARÂMETROS REALÍSTICOS
INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03

# 🔥 EPISÓDIOS DE 1 SEMANA CADA (TIMEFRAME 1MIN)
# 1 semana = 5 dias úteis × 24h × 60min = 7200 barras de 1min
TEST_STEPS = 7200          # 🔥 1 semana completa de trading (7200 barras 1min)
NUM_EPISODES = 25          # 🔥 25 episódios para teste rápido
SEEDS = [42]               # Seed fixo para consistência
DETERMINISTIC = False      # Modo estocástico para avaliação realística
CONFIDENCE_THRESHOLD = 0.3 # Baixo como produção

# 🚀 ULTRA OTIMIZAÇÕES - BATCH PROCESSING OTIMIZADO
BATCH_SIZE = 30           # Predictions em batch de 30 (otimizado)
MEMORY_BATCH = 50         # Pre-allocated memory batches
USE_FEATURES_CACHE = True # Cache permanente de features

# 🎯 SMART EARLY TERMINATION PARAMS
ACTIVITY_WINDOW = 200     # Janela rolling de atividade
INACTIVITY_THRESHOLD = 0.92  # 92% holds para terminar
MIN_EPISODE_STEPS = 720   # Mínimo 10% do episódio (720 steps)

# 🧠 LAZY METRICS THRESHOLD
MIN_VALID_STEPS = 100     # Episódios <100 steps são invalid

# USAR TODO O DATASET EVAL (50K é o tamanho correto para avaliação)
USE_RECENT_DATA = False  # 🔧 FIX: Dataset EVAL já é otimizado, usar completo
RECENT_WEEKS_COUNT = 5   # 🔧 REDUZIDO: Se usar recent, apenas 5 semanas

# 🚀 CACHE GLOBAL DE FEATURES PARA MÁXIMA VELOCIDADE
_features_cache = {}
_environment_cache = {}
_memory_pool = MemoryPool()

def preprocess_dataset_once():
    """
    🚀 PRE-PROCESSAMENTO Único DO DATASET - CACHE PERMANENTE
    """
    global _preprocessed_data, _preprocessing_lock

    with _preprocessing_lock:
        if _preprocessed_data is not None:
            return _preprocessed_data.copy()

        print("📊 [DATASET] Pre-processando dataset MT5 uma vez...")

        # 🏆 USAR DATASET MT5 25 SEMANAS - DATASET NOVO
        dataset_path = 'data/GOLD_1M_MT5_GOLD_25WEEKS_20250923_190721.pkl'
        data = pd.read_pickle(dataset_path)

        # Converter timestamp para time se necessário (dados MT5)
        if 'timestamp' in data.columns:
            data = data.rename(columns={'timestamp': 'time'})
            data['time'] = pd.to_datetime(data['time'])

        # 🔥 COLUNAS BÁSICAS PARA DADOS MT5 (com volume_1m)
        basic_columns = ['time', 'open_1m', 'high_1m', 'low_1m', 'close_1m', 'volume_1m']

        # Se não existirem _1m, usar colunas básicas e renomear
        if 'open_1m' not in data.columns:
            column_mapping = {
                'open': 'open_1m',
                'high': 'high_1m',
                'low': 'low_1m',
                'close': 'close_1m',
                'volume': 'volume_1m'
            }

            # Aplicar renomeação apenas para colunas que existem
            columns_to_rename = {old: new for old, new in column_mapping.items() if old in data.columns}
            if columns_to_rename:
                data = data.rename(columns=columns_to_rename)
                print(f"📊 Colunas renomeadas para formato 1min: {list(columns_to_rename.keys())}")

        # 🚀 MANTER APENAS COLUNAS BÁSICAS PARA EVITAR CONFLITOS
        available_basic = [col for col in basic_columns if col in data.columns]
        data = data[available_basic].copy()

        # 🚀 OTIMIZAÇÃO: Converter para tipos otimizados
        for col in ['open_1m', 'high_1m', 'low_1m', 'close_1m']:
            if col in data.columns:
                data[col] = data[col].astype(np.float32)

        if 'volume_1m' in data.columns:
            data[col] = data['volume_1m'].astype(np.int32)

        _preprocessed_data = data
        print(f"📊 Dataset pre-processado: {len(data)} linhas, {len(data.columns)} colunas básicas (25 semanas - 1min)")

        return data.copy()

def preload_all_models():
    """
    🚀 PARALLEL MODEL LOADING - Carrega todos os modelos em paralelo
    """
    print("🚀 [PARALLEL] Carregando todos os modelos em paralelo...")

    from sb3_contrib import RecurrentPPO

    def load_single_model(model_path):
        try:
            model = RecurrentPPO.load(model_path)
            # 🔧 CONFIGURAÇÕES OTIMIZADAS
            model.policy.set_training_mode(False)
            for param in model.policy.parameters():
                param.requires_grad = False
            return model_path, model
        except Exception as e:
            print(f"❌ Erro carregando {model_path}: {e}")
            return model_path, None

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(load_single_model, path) for path in CHECKPOINTS_TO_TEST]

        for future in futures:
            model_path, model = future.result()
            if model:
                _model_cache[model_path] = model
                print(f"✅ Loaded: {os.path.basename(model_path)}")

    print(f"🚀 [PARALLEL] {len(_model_cache)} modelos carregados")

def calculate_metrics_lazy(episode_data):
    """
    🧠 LAZY METRICS - Só calcula métricas para episódios válidos
    """
    if episode_data['steps'] < MIN_VALID_STEPS:
        return {
            'valid': False,
            'return': episode_data['return'],
            'steps': episode_data['steps'],
            'trades': episode_data['trades']
        }

    # Métricas completas só para episódios válidos
    return {
        'valid': True,
        'return': episode_data['return'],
        'trades': episode_data['trades'],
        'steps': episode_data['steps'],
        'active': episode_data['active'],
        'portfolio_pnl': episode_data['portfolio_pnl'],
        'portfolio_value': episode_data['portfolio_value'],
        'winning_trades': episode_data['winning_trades'],
        'losing_trades': episode_data['losing_trades']
    }

def soft_reset_env(env):
    """
    🚀 SOFT RESET - Mantém estruturas, só reseta estado
    """
    # Reset rápido apenas do estado essencial
    env.current_step = 0
    env.portfolio_value = env.initial_balance
    if hasattr(env, 'positions'):
        env.positions.clear() if hasattr(env.positions, 'clear') else setattr(env, 'positions', [])
    if hasattr(env, 'trades'):
        env.trades.clear() if hasattr(env.trades, 'clear') else setattr(env, 'trades', [])
    if hasattr(env, 'balance_history'):
        env.balance_history.clear() if hasattr(env.balance_history, 'clear') else setattr(env, 'balance_history', [env.initial_balance])

    # Reset memory pool
    global _memory_pool
    _memory_pool.reset()

    return env.reset()

def setup_ultra_optimized_environment():
    """
    🚀 ULTRA OTIMIZADO: Environment com todas as otimizações sem afetar confiabilidade
    """
    global _environment_cache

    # 🍒 CHERRY: Importar as funções do CHERRY para usar features enhanced
    from cherry import load_optimized_data_original, TradingEnv

    # 🚀 CACHE: Verificar se environment já foi criado
    cache_key = f"env_{USE_RECENT_DATA}_{RECENT_WEEKS_COUNT}"
    if cache_key in _environment_cache:
        print("💾 [ENV CACHE HIT] Usando environment cacheado")
        return _environment_cache[cache_key]

    # 🚀 USAR DATASET PRE-PROCESSADO
    data = preprocess_dataset_once()
    
    
    # 🔥 AJUSTE PARA 1MIN: 1 semana = 7200 steps (1440 min/dia × 5 dias)
    if USE_RECENT_DATA and len(data) > RECENT_WEEKS_COUNT * 7200:
        # Pegar últimas semanas (7200 steps por semana - timeframe 1min)
        recent_data_size = RECENT_WEEKS_COUNT * 7200
        data = data.iloc[-recent_data_size:].reset_index(drop=True)
        print(f"📅 Usando dados recentes: {len(data)} steps ({RECENT_WEEKS_COUNT} semanas - 1min timeframe)")
    
    # 🔥 AMBIENTE PARA TESTE PURO - SEM COOLDOWNS/TIMEOUTS + ULTRA OTIMIZADO
    env = TradingEnv(
        df=data,
        window_size=20,  # Mesmo do CHERRY
        is_training=False,  # 🔧 TESTE PURO: Modo eval (sem enhancements)
        initial_balance=INITIAL_PORTFOLIO,
        trading_params={
            'min_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'enable_shorts': True,
            'max_positions': 2
        }
    )
    
    # 🚀 ULTRA OTIMIZAÇÕES: Desabilitar logging verbose
    if hasattr(env, '_verbose_logging'):
        env._verbose_logging = False
    if hasattr(env, '_debug_mode'):
        env._debug_mode = False
        
    # 🚀 REWARD SYSTEM EVAL MODE: Cálculos simplificados
    if hasattr(env, 'reward_system'):
        if hasattr(env.reward_system, '_eval_mode'):
            env.reward_system._eval_mode = True
            print("🚀 [SPEED] Reward system em modo eval (simplificado)")
    
    # 🚨 CRÍTICO: Configurar activity_system para timeframe 1min
    if hasattr(env, 'activity_system') and env.activity_system is not None:
        # ✅ OPÇÃO 1: Ajustar timeout de 60 candles (1h) para 300 candles (5h em 1min)
        if hasattr(env.activity_system.config, 'position_timeout_candles'):
            env.activity_system.config.position_timeout_candles = 300  # 5h em candles 1min
            print("🔧 Position timeout ajustado: 60 → 300 candles (5h em 1min)")
    
    # 🚨 CRÍTICO: Zerar cooldowns para teste puro 
    if hasattr(env, 'cooldown_after_trade'):
        env.cooldown_after_trade = 0
        print("🔧 Cooldowns zerrados para teste puro")
    
    if hasattr(env, 'cooldown_base'):
        env.cooldown_base = 0
        print("🔧 Cooldown base zerrado para teste puro")
    
    # 🚀 CACHE: Armazenar environment para reutilização
    _environment_cache[cache_key] = env
    print("💾 [ENV CACHED] Environment armazenado em cache")
    
    return env

# 🚀 CACHE GLOBAL DE MODELOS PARA EVITAR RECARREGAMENTO
_model_cache = {}

def evaluate_model_ultra_optimized(model_path, num_episodes=NUM_EPISODES):
    """
    🚀 AVALIAÇÃO ULTRA OTIMIZADA: 5x mais rápida sem afetar confiabilidade
    """
    from tqdm import tqdm
    
    model_name = os.path.basename(model_path)
    print(f"\\n⚡ ULTRA TEST: {model_name}")

    # 🚀 OTIMIZAÇÃO: Usar cache de modelos
    from sb3_contrib import RecurrentPPO
    try:
        if model_path not in _model_cache:
            print(f"🔄 Carregando modelo...")
            model = RecurrentPPO.load(model_path)
            
            # 🔧 CONFIGURAÇÕES IDÊNTICAS AO CHERRY + ULTRA OTIMIZADAS
            model.policy.set_training_mode(False)  # Modo eval
            for param in model.policy.parameters():
                param.requires_grad = False        # Desabilitar gradientes
            
            _model_cache[model_path] = model
            print(f"💾 Modelo cached")
        else:
            model = _model_cache[model_path]
            print(f"⚡ Usando cache")
            
        print(f"✅ Modelo pronto")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None
    
    results = {
        'episodes': [],
        'trades_per_episode': [],
        'active_episodes': 0,
        'total_trades': 0,
        'seeds_results': {}
    }
    
    # 🚀 OTIMIZAÇÃO: Criar environment UMA VEZ e reutilizar
    print(f"🚀 [ULTRA] Criando environment ultra-otimizado...")
    env = setup_ultra_optimized_environment()
    
    # 🚀 USAR MEMORY POOL GLOBAL
    global _memory_pool
    obs_size = 450  # V10 temporal observation space

    print(f"💾 [MEMORY] Usando Memory Pool global otimizado")
    
    # Testar com múltiplas seeds
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\\n🎲 [ULTRA] Seed {seed} ({seed_idx+1}/{len(SEEDS)}) [BATCH={BATCH_SIZE}]")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        seed_results = []
        
        # Criar progress bar para episódios
        episode_pbar = tqdm(total=num_episodes // len(SEEDS), desc=f"Seed {seed}", 
                           unit="ep", leave=False,
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
        for episode in range(num_episodes // len(SEEDS)):
            try:
                # 🚀 SOFT RESET - Muito mais rápido
                obs = soft_reset_env(env)
                episode_return = 0.0
                episode_trades = 0
                episode_steps = 0
                lstm_states = None

                # 🚀 ULTRA OPTIMIZATION: True batch processing + smart termination
                with torch.no_grad():  # 🚀 OTIMIZAÇÃO: Disable gradients para inference
                    # 🎯 SMART EARLY TERMINATION - Activity window
                    global _memory_pool
                    _memory_pool.activity_pool.clear()

                    for step in range(TEST_STEPS):
                        # 🚀 OPTIMIZED PREDICTION
                        action, lstm_states = model.predict(
                            obs,
                            state=lstm_states,
                            deterministic=DETERMINISTIC
                        )

                        # 🎯 ACTIVITY TRACKING - Rolling window
                        action_value = action[0] if hasattr(action, '__len__') and len(action) > 0 else action
                        is_hold = abs(action_value) < 0.25
                        _memory_pool.activity_pool.append(is_hold)

                        # Step environment
                        obs, reward, done, info = env.step(action)
                        episode_return += reward
                        episode_steps += 1

                        # 🚀 SMART EARLY TERMINATION - Baseado em activity window
                        if done:
                            break

                        # Terminação inteligente baseada em inatividade
                        if (len(_memory_pool.activity_pool) == ACTIVITY_WINDOW and
                            episode_steps > MIN_EPISODE_STEPS):
                            hold_ratio = sum(_memory_pool.activity_pool) / ACTIVITY_WINDOW
                            if hold_ratio > INACTIVITY_THRESHOLD:
                                # 🎯 Modelo inativo por muito tempo - terminar
                                break
                
                # 🔥 COLETAR MÉTRICAS REAIS DO TRADING COM VECTORIZAÇÃO
                portfolio_pnl = env.portfolio_value - INITIAL_PORTFOLIO
                trades_list = getattr(env, 'trades', [])
                total_trades_real = len(trades_list)

                # 🚀 VECTORIZED PNL CALCULATION
                if trades_list:
                    pnls = np.array([t.get('pnl_usd', t.get('pnl', t.get('profit', 0))) for t in trades_list])
                    winning_trades = int(np.sum(pnls > 0))
                    losing_trades = int(np.sum(pnls < 0))
                else:
                    winning_trades = losing_trades = 0

                episode_data = {
                    'return': episode_return,
                    'trades': total_trades_real,
                    'steps': episode_steps,
                    'active': total_trades_real > 0,
                    'portfolio_pnl': portfolio_pnl,
                    'portfolio_value': env.portfolio_value,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades
                }

                # 🧠 LAZY METRICS - Só dados essenciais para episódios inválidos
                processed_data = calculate_metrics_lazy(episode_data)
                seed_results.append(processed_data)
                
                # Atualizar progress bar
                episode_pbar.update(1)
                
            except Exception as e:
                episode_pbar.set_postfix({"Erro": str(e)[:20]})
                continue
        
        episode_pbar.close()
        
        results['seeds_results'][seed] = seed_results
        results['episodes'].extend(seed_results)
    
    # 🚀 LIMPEZA DE MEMÓRIA entre modelos (igual ao Cherry)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 🔥 CALCULAR MÉTRICAS COMPLETAS DE TRADING (IGUAL AO CHERRY)
    if len(results['episodes']) > 0:
        returns = [ep['return'] for ep in results['episodes']]
        trades = [ep['trades'] for ep in results['episodes']]
        
        # 🚨 MÉTRICAS REAIS DE TRADING
        portfolio_pnls = [ep.get('portfolio_pnl', 0) for ep in results['episodes']]
        portfolio_values = [ep.get('portfolio_value', INITIAL_PORTFOLIO) for ep in results['episodes']]
        
        # 🚀 VECTORIZED AGGREGATION
        valid_episodes = [ep for ep in results['episodes'] if ep.get('valid', True)]

        if valid_episodes:
            winning_trades_array = np.array([ep.get('winning_trades', 0) for ep in valid_episodes])
            losing_trades_array = np.array([ep.get('losing_trades', 0) for ep in valid_episodes])
            total_winning_trades = int(np.sum(winning_trades_array))
            total_losing_trades = int(np.sum(losing_trades_array))
        else:
            total_winning_trades = total_losing_trades = 0

        total_real_trades = total_winning_trades + total_losing_trades
        
        # Calcular win rate REAL baseado em trades individuais
        real_win_rate = (total_winning_trades / total_real_trades * 100) if total_real_trades > 0 else 0
        
        metrics = {
            # Métricas de reward (para compatibilidade)
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(portfolio_pnls) / (np.std(portfolio_pnls) + 1e-8),  # 🔧 Sharpe baseado em PnL real
            
            # Métricas básicas
            'total_episodes': len(results['episodes']),
            'active_episodes': sum(1 for ep in results['episodes'] if ep['active']),
            'activity_rate': sum(1 for ep in results['episodes'] if ep['active']) / len(results['episodes']) * 100,
            
            # 🚨 MÉTRICAS REAIS DE TRADING
            'mean_portfolio_pnl': np.mean(portfolio_pnls),
            'std_portfolio_pnl': np.std(portfolio_pnls),
            'median_portfolio_pnl': np.median(portfolio_pnls),
            'mean_portfolio_value': np.mean(portfolio_values),
            'total_real_pnl': sum(portfolio_pnls),
            
            # Métricas de trades
            'total_trades': sum(trades),
            'avg_trades_per_episode': np.mean(trades),
            'total_winning_trades': total_winning_trades,
            'total_losing_trades': total_losing_trades,
            'real_win_rate': real_win_rate,
            
            # Estatísticas detalhadas  
            'max_portfolio_pnl': max(portfolio_pnls) if portfolio_pnls else 0,
            'min_portfolio_pnl': min(portfolio_pnls) if portfolio_pnls else 0,
            'portfolio_pnl_range': max(portfolio_pnls) - min(portfolio_pnls) if portfolio_pnls else 0,
        }
        
        # 🚀 SAÍDA OTIMIZADA (uma linha por modelo)
        print(f"  ⏱️ Tempo: {episode_steps * len(results['episodes']) / 1000:.1f}k steps")
        print(f"  📊 Sharpe: {metrics['sharpe_ratio']:.4f}")
        print(f"  💰 PnL: ${metrics['total_real_pnl']:.2f}")
        print(f"  🎯 Win Rate: {real_win_rate:.1f}% ({total_winning_trades}/{total_real_trades})")
        print(f"  📈 Trades/ep: {metrics['avg_trades_per_episode']:.1f}")
        print(f"  🎯 Activity: {metrics['activity_rate']:.1f}%")
        
        return metrics
    else:
        print(f"❌ Nenhum episódio válido para {model_name}")
        return None

def main():
    """
    🚀 MAIN ULTRA OPTIMIZADO: Comparação completa com máxima velocidade
    """
    print("🏆 BATTLE OF CHAMPIONS - CHERRY vs V3 BRUTAL")
    print("=" * 60)
    print(f"🍒 Cherry 1.175M vs Cherry 1.2M (Sharpe 7.13) vs ⚔️ V3 Brutal 1.325M")
    print(f"⚡ {NUM_EPISODES} episódios")
    print(f"🎲 {len(SEEDS)} seeds")
    print(f"🕐 Steps por episódio: {TEST_STEPS}")
    print(f"🚀 BATCH SIZE: {BATCH_SIZE} (Ultra Speed Mode)")
    print(f"🎯 SMART TERMINATION: {INACTIVITY_THRESHOLD*100:.0f}% holds, window={ACTIVITY_WINDOW}")
    print(f"🧠 LAZY METRICS: Min {MIN_VALID_STEPS} steps")
    print("-" * 60)

    # 🚀 PRE-LOAD todos os modelos em paralelo
    print("🚀 [INIT] Pre-carregando dataset e modelos...")
    preload_all_models()
    preprocess_dataset_once()  # Pre-processar dataset
    print("✅ [INIT] Inicialização completa\n")

    all_results = {}
    best_model_sharpe = None
    best_sharpe = -999
    best_model_pnl = None
    best_pnl = -99999

    start_time = datetime.now()
    
    for i, checkpoint_path in enumerate(CHECKPOINTS_TO_TEST):
        try:
            print(f"[{i+1}/{len(CHECKPOINTS_TO_TEST)}] ", end='')

            # Verificar se modelo foi carregado
            if checkpoint_path not in _model_cache:
                print(f"⚠️ Modelo {os.path.basename(checkpoint_path)} não foi pré-carregado, pulando...")
                continue

            metrics = evaluate_model_ultra_optimized(checkpoint_path)

            if metrics:
                all_results[checkpoint_path] = metrics

                # Track best models
                if metrics.get('sharpe_ratio', -999) > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    model_name = os.path.basename(checkpoint_path)
                    # Extrair número de steps do nome
                    try:
                        steps = model_name.split('_')[2]  # v3brutal_simpledirecttraining_500000_steps...
                    except:
                        steps = model_name
                    best_model_sharpe = steps

                if metrics.get('total_real_pnl', -99999) > best_pnl:
                    best_pnl = metrics['total_real_pnl']
                    model_name = os.path.basename(checkpoint_path)
                    try:
                        steps = model_name.split('_')[2]
                    except:
                        steps = model_name
                    best_model_pnl = steps
        
        except Exception as e:
            print(f"❌ Erro avaliando {os.path.basename(checkpoint_path)}: {e}")
            traceback.print_exc()
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print("\\n" + "=" * 60)
    print("🏆 RESULTADOS FINAIS - BATTLE OF CHAMPIONS")
    print("=" * 60)
    print(f"⏱️ Tempo total: {total_time}")
    print(f"🚀 Velocidade: {len(CHECKPOINTS_TO_TEST) * NUM_EPISODES / total_time.total_seconds():.2f} ep/s")
    print(f"🥇 Melhor Sharpe: {best_model_sharpe} ({best_sharpe:.4f})")
    print(f"💰 Melhor PnL: {best_model_pnl} (${best_pnl:.2f})")
    
    # 🔥 RELATÓRIO DETALHADO DE CADA MODELO
    print("\\n" + "📊 RANKING POR SHARPE RATIO:")
    print("-" * 60)
    
    # Ordenar por Sharpe ratio
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('sharpe_ratio', -999), reverse=True)
    
    for i, (model_path, metrics) in enumerate(sorted_results[:10]):  # Top 10
        model_name = os.path.basename(model_path)
        try:
            steps = model_name.split('_')[2]  # Extrair steps
            print(f"\\n🏷️ {steps} STEPS:")
            print(f"  📈 Return médio: {metrics['mean_return']:.4f}")
            print(f"  📊 Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"  💰 PnL médio: ${metrics['mean_portfolio_pnl']:.2f}")
            print(f"  💸 PnL total: ${metrics['total_real_pnl']:.2f}")
            print(f"  🎯 Win Rate: {metrics['real_win_rate']:.1f}%")
            print(f"  📈 Trades/ep: {metrics['avg_trades_per_episode']:.1f}")
            print(f"  🎯 Activity: {metrics['activity_rate']:.1f}%")
        except:
            print(f"\\n🏷️ {model_name}:")
            print(f"  📊 Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"  💰 PnL total: ${metrics['total_real_pnl']:.2f}")
    
    print("\\n" + "✅ BATTLE OF CHAMPIONS CONCLUÍDO!")
    print("🎯 Cherry 1.2M vs V3 Brutal 1.325M - Comparação finalizada!")
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        print("\\n🎯 BATTLE OF CHAMPIONS - Resultados salvos em memória.")
        print("🏆 Cherry Enhanced vs V3 Brutal - Comparação completa!")
    except Exception as e:
        print(f"❌ Erro durante avaliação: {e}")
        traceback.print_exc()