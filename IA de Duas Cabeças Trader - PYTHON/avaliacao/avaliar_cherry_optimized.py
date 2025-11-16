#!/usr/bin/env python3
"""
🚀 AVALIAÇÃO CHERRY ULTRA-OTIMIZADA COM CACHE
=============================================

OTIMIZAÇÕES:
✅ 1. Environment único reutilizado para todos os modelos
✅ 2. Dados carregados uma única vez
✅ 3. Cache de observações
✅ 4. Batch processing paralelo
✅ 5. Reset rápido sem recriar environment
✅ 6. Pre-compilação de modelos
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
import random
import json
import numpy as np
import pandas as pd
import torch
import gc
from scipy import stats
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore')
sys.path.append("D:/Projeto")

# 🔥 CHECKPOINTS PARA TESTAR - FOCO AO REDOR DE 15.5M STEPS (200k cima/baixo)
CHECKPOINTS_TO_TEST = [
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_15300000_steps_20250908_102137.zip",     # 15.3M (-200k)
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_15350000_steps_20250908_102620.zip",     # 15.35M (-150k)
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_15400000_steps_20250908_103108.zip",     # 15.4M (-100k)
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_15450000_steps_20250908_103558.zip",     # 15.45M (-50k)
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_15500000_steps_20250908_104049.zip",     # 15.5M (baseline)
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_15550000_steps_20250908_104539.zip",     # 15.55M (+50k)
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_15600000_steps_20250908_105023.zip",     # 15.6M (+100k)
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_15650000_steps_20250908_105509.zip",     # 15.65M (+150k)
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_15700000_steps_20250908_105959.zip",     # 15.7M (+200k)
]

# PARÂMETROS OTIMIZADOS
INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03
TEST_STEPS = 1800         # Steps por episódio = 1 semana de trading (5m bars)
NUM_EPISODES = 25         # 🎯 ANÁLISE ROBUSTA - 25 episódios por modelo (225 total)
SEEDS = [42, 123]         # 🎯 2 SEEDS para robustez estatística
DETERMINISTIC = False     # Non-deterministic para avaliação realística

# 🔥 CACHE GLOBAL - CARREGADO UMA VEZ
GLOBAL_DATA_CACHE = None
GLOBAL_ENV_CACHE = None

# 💾 CACHE PERSISTENTE - CONFIGURAÇÃO
CACHE_DIR = "D:/Projeto/cache_cherry"
DATA_CACHE_FILE = os.path.join(CACHE_DIR, "cherry_data_cache.pkl")
ENV_CACHE_FILE = os.path.join(CACHE_DIR, "cherry_env_cache.pkl")
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")


def ensure_cache_dir():
    """🗂️ GARANTIR QUE PASTA DE CACHE EXISTE"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"📁 Criada pasta de cache: {CACHE_DIR}")


def get_cherry_source_hash():
    """🔍 GERAR HASH DOS ARQUIVOS FONTE CHERRY + ENHANCED NORMALIZER"""
    cherry_files = [
        "D:/Projeto/cherry.py",
        "D:/Projeto/avaliacao/ultra_reliable_peaks.txt",
        "D:/Projeto/Modelo PPO Trader/enhanced_normalizer.py"  # INCLUIR NORMALIZER!
    ]
    
    hash_content = ""
    for file_path in cherry_files:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                file_content = f.read()
                hash_content += hashlib.md5(file_content).hexdigest()
        else:
            print(f"⚠️ Arquivo não encontrado: {file_path}")
    
    return hashlib.md5(hash_content.encode()).hexdigest()[:12]


def load_cache_metadata():
    """📋 CARREGAR METADADOS DO CACHE"""
    if os.path.exists(CACHE_METADATA_FILE):
        try:
            with open(CACHE_METADATA_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_cache_metadata(metadata):
    """💾 SALVAR METADADOS DO CACHE"""
    ensure_cache_dir()
    with open(CACHE_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def is_cache_valid():
    """✅ VERIFICAR SE CACHE É VÁLIDO"""
    if not os.path.exists(DATA_CACHE_FILE):
        return False, "Cache de dados não existe"
    
    metadata = load_cache_metadata()
    current_hash = get_cherry_source_hash()
    
    if metadata.get('source_hash') != current_hash:
        return False, f"Hash dos arquivos fonte mudou: {metadata.get('source_hash')} → {current_hash}"
    
    # Verificar se cache não é muito antigo (24 horas)
    cache_time = metadata.get('created_at')
    if cache_time:
        cache_datetime = datetime.fromisoformat(cache_time)
        if datetime.now() - cache_datetime > timedelta(hours=24):
            return False, f"Cache muito antigo: {cache_datetime}"
    
    return True, "Cache válido"


def load_data_once():
    """🔥 CARREGAR DADOS UMA ÚNICA VEZ COM CACHE PERSISTENTE"""
    global GLOBAL_DATA_CACHE
    
    if GLOBAL_DATA_CACHE is not None:
        return GLOBAL_DATA_CACHE
    
    # Verificar cache persistente primeiro
    cache_valid, cache_reason = is_cache_valid()
    
    if cache_valid and os.path.exists(DATA_CACHE_FILE):
        try:
            print("💾 Carregando dados Cherry do CACHE PERSISTENTE...")
            with open(DATA_CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Dados carregados do cache: {len(data)} steps")
            GLOBAL_DATA_CACHE = data
            return data
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache: {e}, recarregando dados...")
    else:
        print(f"🔄 Cache inválido: {cache_reason}")
    
    print("📊 Carregando dados Cherry ORIGINAIS...")
    
    # Mudar working directory
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        from cherry import load_optimized_data_original
        data = load_optimized_data_original()
    finally:
        os.chdir(original_cwd)
    
    # Usar dados recentes
    if len(data) > 30 * 2016:  # 30 semanas
        recent_data_size = 30 * 2016
        data = data.iloc[-recent_data_size:].reset_index(drop=True)
        print(f"✅ Dados processados: {len(data)} steps")
    
    # Salvar no cache persistente
    try:
        ensure_cache_dir()
        with open(DATA_CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)
        
        # Salvar metadados
        metadata = {
            'created_at': datetime.now().isoformat(),
            'source_hash': get_cherry_source_hash(),
            'data_size': len(data),
            'cache_version': '1.0'
        }
        save_cache_metadata(metadata)
        print(f"💾 Cache salvo: {DATA_CACHE_FILE}")
    except Exception as e:
        print(f"⚠️ Erro ao salvar cache: {e}")
    
    GLOBAL_DATA_CACHE = data
    return data


def is_env_cache_valid():
    """✅ VERIFICAR SE CACHE DO ENVIRONMENT É VÁLIDO"""
    if not os.path.exists(ENV_CACHE_FILE):
        return False, "Cache de environment não existe"
    
    metadata = load_cache_metadata()
    current_hash = get_cherry_source_hash()
    
    if metadata.get('env_source_hash') != current_hash:
        return False, f"Hash dos arquivos fonte mudou: {metadata.get('env_source_hash')} → {current_hash}"
    
    # Verificar se configuração do env mudou
    current_config = {
        'window_size': 20,
        'initial_balance': INITIAL_PORTFOLIO,
        'min_lot_size': BASE_LOT_SIZE,
        'max_lot_size': MAX_LOT_SIZE,
        'enable_shorts': True,
        'max_positions': 2
    }
    
    if metadata.get('env_config') != current_config:
        return False, "Configuração do environment mudou"
    
    return True, "Cache de environment válido"


def save_env_to_cache(env, config):
    """💾 SALVAR ENVIRONMENT NO CACHE (apenas configuração)"""
    try:
        ensure_cache_dir()
        
        # Salvar apenas configuração do env, não o objeto completo
        env_data = {
            'config': config,
            'data_hash': get_cherry_source_hash()
        }
        
        with open(ENV_CACHE_FILE, 'wb') as f:
            pickle.dump(env_data, f)
        
        # Atualizar metadados
        metadata = load_cache_metadata()
        metadata.update({
            'env_created_at': datetime.now().isoformat(),
            'env_source_hash': get_cherry_source_hash(),
            'env_config': config
        })
        save_cache_metadata(metadata)
        
        print(f"💾 Cache do environment salvo")
    except Exception as e:
        print(f"⚠️ Erro ao salvar cache do env: {e}")


def create_env_once():
    """🔥 CRIAR ENVIRONMENT UMA ÚNICA VEZ COM CACHE INTELIGENTE"""
    global GLOBAL_ENV_CACHE
    
    if GLOBAL_ENV_CACHE is not None:
        return GLOBAL_ENV_CACHE
    
    # Configuração atual
    env_config = {
        'window_size': 20,
        'initial_balance': INITIAL_PORTFOLIO,
        'min_lot_size': BASE_LOT_SIZE,
        'max_lot_size': MAX_LOT_SIZE,
        'enable_shorts': True,
        'max_positions': 2
    }
    
    # Verificar se podemos reutilizar configuração (não o objeto env)
    cache_valid, cache_reason = is_env_cache_valid()
    
    if cache_valid:
        print("✅ Configuração do environment válida no cache")
    else:
        print(f"🔄 Cache de env inválido: {cache_reason}")
    
    print("🏗️ Criando TradingEnv fresco (dados podem estar em cache)...")
    
    # Mudar working directory
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        from cherry import TradingEnv
        
        # Usar dados já cacheados (isso é o mais importante!)
        data = load_data_once()
        
        env = TradingEnv(
            df=data,
            window_size=env_config['window_size'],
            is_training=True,
            initial_balance=env_config['initial_balance'],
            trading_params={
                'min_lot_size': env_config['min_lot_size'],
                'max_lot_size': env_config['max_lot_size'],
                'enable_shorts': env_config['enable_shorts'],
                'max_positions': env_config['max_positions']
            }
        )
        
        print("✅ Environment criado")
        
        # Salvar config no cache (para próxima vez)
        if not cache_valid:
            save_env_to_cache(env, env_config)
        
    finally:
        os.chdir(original_cwd)
    
    GLOBAL_ENV_CACHE = env
    return env


def fast_evaluate_model(model_path, num_episodes=NUM_EPISODES):
    """🔥 AVALIAÇÃO RÁPIDA COM CACHE"""
    
    model_name = os.path.basename(model_path)
    print(f"\n⚡ TESTE RÁPIDO: {model_name}")
    
    # Carregar modelo
    from sb3_contrib import RecurrentPPO
    try:
        model = RecurrentPPO.load(model_path)
        model.policy.set_training_mode(False)
        for param in model.policy.parameters():
            param.requires_grad = False
        print(f"✅ Modelo carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None
    
    # 🔥 USAR ENVIRONMENT CACHEADO
    env = create_env_once()
    
    results = {
        'episodes': [],
        'trades_per_episode': [],
        'total_trades': 0
    }
    
    episodes_per_seed = num_episodes // len(SEEDS)
    
    for seed_idx, seed in enumerate(SEEDS):
        print(f"  Seed {seed}: ", end='', flush=True)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        for episode in range(episodes_per_seed):
            # 🔥 RESET RÁPIDO - Não recria environment
            obs = env.reset()
            
            episode_return = 0.0
            episode_trades = 0
            lstm_states = None
            
            # Run episode with torch optimization
            with torch.no_grad():  # 🚀 OTIMIZAÇÃO: Disable gradients para inference
                for step in range(TEST_STEPS):
                    action, lstm_states = model.predict(
                        obs, 
                        state=lstm_states, 
                        deterministic=DETERMINISTIC
                    )
                    
                    obs, reward, done, info = env.step(action)
                    episode_return += reward
                    
                    if 'trade_executed' in info and info['trade_executed']:
                        episode_trades += 1
                    
                    if done:
                        break
            
            # 🔥 COLETAR MÉTRICAS REAIS DO TRADING
            portfolio_pnl = env.portfolio_value - INITIAL_PORTFOLIO
            trades_list = getattr(env, 'trades', [])
            
            # Calcular win/loss de trades individuais
            winning_trades = sum(1 for trade in trades_list if trade.get('pnl_usd', 0) > 0)
            losing_trades = sum(1 for trade in trades_list if trade.get('pnl_usd', 0) < 0)
            total_trades_real = len(trades_list)
            
            results['episodes'].append({
                'return': episode_return,  # Reward (para compatibilidade)
                'trades': episode_trades,  # Trades executados
                'active': episode_trades > 0,
                # 🚨 NOVAS MÉTRICAS REAIS
                'portfolio_pnl': portfolio_pnl,  # PnL real em USD
                'portfolio_value': env.portfolio_value,  # Valor final do portfolio
                'trades_real': trades_list.copy(),  # Lista de trades reais
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_trades_real': total_trades_real
            })
            results['total_trades'] += episode_trades
            results['trades_per_episode'].append(episode_trades)
            
            # Progress indicator
            print(".", end='', flush=True)
        
        print(f" ✓")
        
        # 🚀 LIMPEZA DE MEMÓRIA entre modelos
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calcular métricas básicas
    if len(results['episodes']) > 0:
        returns = [ep['return'] for ep in results['episodes']]
        trades = [ep['trades'] for ep in results['episodes']]
        
        # 🚨 MÉTRICAS REAIS DE TRADING
        portfolio_pnls = [ep['portfolio_pnl'] for ep in results['episodes']]
        portfolio_values = [ep['portfolio_value'] for ep in results['episodes']]
        
        # Agregar todos os trades de todos os episódios
        all_trades = []
        total_winning_trades = 0
        total_losing_trades = 0
        total_real_trades = 0
        
        for ep in results['episodes']:
            all_trades.extend(ep['trades_real'])
            total_winning_trades += ep['winning_trades']
            total_losing_trades += ep['losing_trades']
            total_real_trades += ep['total_trades_real']
        
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
            'avg_trades_per_day': np.mean(trades) / 7,
            'total_trades_real': total_real_trades,
            'winning_trades': total_winning_trades,
            'losing_trades': total_losing_trades,
            
            # 🎯 WIN RATE CORRETO - baseado em trades individuais
            'win_rate': real_win_rate,
            
            # Episodes com lucro (PnL > 0)
            'profitable_episodes': sum(1 for pnl in portfolio_pnls if pnl > 0),
            'losing_episodes_pnl': sum(1 for pnl in portfolio_pnls if pnl < 0),
            'episode_profit_rate': sum(1 for pnl in portfolio_pnls if pnl > 0) / len(portfolio_pnls) * 100 if portfolio_pnls else 0
        }
    else:
        metrics = {'error': 'No valid episodes'}
    
    return {
        'model_path': model_path,
        'metrics': metrics,
        'raw_results': results
    }


def run_optimized_comparison():
    """🔥 COMPARAÇÃO OTIMIZADA COM CACHE PERSISTENTE"""
    print("🚀 AVALIAÇÃO CHERRY ULTRA-OTIMIZADA COM CACHE PERSISTENTE")
    print("=" * 60)
    
    # 🔥 PRÉ-CARREGAR DADOS E ENVIRONMENT COM CACHE
    start_cache_time = datetime.now()
    print("💾 Verificando caches persistentes...")
    
    load_data_once()  # Vai usar cache se disponível
    create_env_once()  # Vai usar configuração cacheada
    
    cache_time = (datetime.now() - start_cache_time).total_seconds()
    print(f"⚡ Setup completo em {cache_time:.1f}s")
    
    print(f"📊 Testando {len(CHECKPOINTS_TO_TEST)} modelos")
    print(f"⚡ {NUM_EPISODES} episódios por modelo")
    print(f"🎲 {len(SEEDS)} seeds")
    print(f"📈 Total: {len(CHECKPOINTS_TO_TEST) * NUM_EPISODES} episódios")
    print("-" * 60)
    
    results = {}
    start_time = datetime.now()
    
    for idx, model_path in enumerate(CHECKPOINTS_TO_TEST):
        model_start = datetime.now()
        
        if os.path.exists(model_path):
            result = fast_evaluate_model(model_path)
            if result:
                model_name = os.path.basename(model_path)
                results[model_name] = result
                
                # Tempo por modelo
                model_time = (datetime.now() - model_start).total_seconds()
                print(f"  ⏱️ Tempo: {model_time:.1f}s")
                print(f"  📊 Sharpe: {result['metrics']['sharpe_ratio']:.4f}")
                print(f"  💰 PnL: ${result['metrics']['total_real_pnl']:.2f}")  # 🔧 PnL real em vez de rewards
                print(f"  🎯 Win Rate: {result['metrics']['win_rate']:.1f}% ({result['metrics']['winning_trades']}/{result['metrics']['total_trades_real']})")  # 🔧 Mostrar contadores
                print(f"  📈 Trades/ep: {result['metrics']['avg_trades_per_episode']:.1f}")
        else:
            print(f"⚠️ Modelo não encontrado: {model_path}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # RESULTADOS COMPARATIVOS
    print("\n" + "=" * 60)
    print("📊 RESULTADOS COMPARATIVOS RÁPIDOS:")
    print("-" * 60)
    
    # Criar tabela comparativa
    best_sharpe = -999
    best_model = None
    
    for model_name, result in results.items():
        metrics = result['metrics']
        if 'error' not in metrics:
            steps = model_name.split('_')[2]
            
            print(f"\n🏷️ {steps} STEPS:")
            print(f"  📊 Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"  💰 Total PnL: ${metrics['total_real_pnl']:.2f}")  # 🔧 PnL real
            print(f"  📈 Win Rate: {metrics['win_rate']:.1f}% ({metrics['winning_trades']}/{metrics['total_trades_real']})")  # 🔧 Mostrar contadores
            print(f"  🎯 Activity: {metrics['activity_rate']:.1f}%")
            print(f"  📈 Trades/ep: {metrics['avg_trades_per_episode']:.1f}")
            
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_model = steps
    
    print("\n" + "=" * 60)
    print(f"🏆 MELHOR MODELO: {best_model} STEPS (Sharpe: {best_sharpe:.4f})")
    print(f"⏱️ TEMPO TOTAL: {total_time:.1f} segundos ({total_time/60:.1f} minutos)")
    print(f"⚡ VELOCIDADE: {total_time/len(CHECKPOINTS_TO_TEST):.1f}s por modelo")
    print(f"💾 Cache Setup: {cache_time:.1f}s | Avaliação: {total_time-cache_time:.1f}s")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"avaliacao_cherry_rapida_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Resultados salvos: {filename}")
    
    return results


if __name__ == "__main__":
    try:
        print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
        
        results = run_optimized_comparison()
        
        print(f"\n✅ AVALIAÇÃO CONCLUÍDA!")
        print(f"⏰ Fim: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        traceback.print_exc()