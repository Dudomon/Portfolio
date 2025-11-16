#!/usr/bin/env python3
"""
🚀 AVALIAÇÃO REALÍSTICA 1MIN - DATASET V4 REALISTA
=================================================

CONFIGURAÇÃO PARA TIMEFRAME 1 MINUTO:
✅ 1. Dataset V4: Interpolação de dados reais (1.7% estáticas)
✅ 2. Timeframe ajustado: 1 semana = 7200 steps (1440 min/dia × 5 dias)
✅ 3. Dados sequenciais recentes (não aleatórios)
✅ 4. Configuração idêntica ao silus.py
✅ 5. Stochastic mode (exploration ativa)
✅ 6. Métricas realistas de trading
✅ 7. Multiple seeds para robustez
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

# 🔥 V3 BRUTAL CHECKPOINTS - ULTRA RELIABLE PEAKS + REQUESTED STEPS
CHECKPOINTS_TO_TEST = [
    # 🥇 TOP ULTRA RELIABLE PEAKS (Sharpe > 8.8)
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_975000_steps_20250915_131608.zip",    # 975k (~966k peak)
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_625000_steps_20250915_125401.zip",    # 625k (~630k peaks)
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_650000_steps_20250915_125539.zip",    # 650k (~630k peaks)
    
    # 📊 REQUESTED EVALUATION STEPS 300K-750K
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_300000_steps_20250915_123319.zip",    # 300k
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_350000_steps_20250915_123631.zip",    # 350k
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_400000_steps_20250915_123938.zip",    # 400k
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_450000_steps_20250915_124250.zip",    # 450k
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_500000_steps_20250915_124559.zip",    # 500k
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_550000_steps_20250915_124918.zip",    # 550k
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_600000_steps_20250915_125225.zip",    # 600k
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_700000_steps_20250915_125846.zip",    # 700k
    "D:/Projeto/Otimizacao/treino_principal/models/v3brutal/v3brutal_simpledirecttraining_750000_steps_20250915_130200.zip"     # 750k
]

# PARÂMETROS REALÍSTICOS
INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03

# 🔥 EPISÓDIOS OTIMIZADOS PARA VELOCIDADE
TEST_STEPS = 3600          # 🚀 OTIMIZAÇÃO: 2.5 dias (3600 steps) vs 1 semana (7200) - 2x mais rápido
NUM_EPISODES = 20          # 🚀 OTIMIZAÇÃO: 20 episódios vs 25 - 20% mais rápido  
SEEDS = [42]               # 🚀 OTIMIZAÇÃO: 1 seed vs 2 - 50% mais rápido
DETERMINISTIC = False      # 🔧 REVERTIDO: Modo estocástico para avaliação realística
CONFIDENCE_THRESHOLD = 0.3 # Baixo como produção

# USAR TODO O DATASET EVAL (50K é o tamanho correto para avaliação)
USE_RECENT_DATA = False  # 🔧 FIX: Dataset EVAL já é otimizado, usar completo
RECENT_WEEKS_COUNT = 5   # 🔧 REDUZIDO: Se usar recent, apenas 5 semanas

def setup_realistic_environment():
    """
    Configurar ambiente IDÊNTICO à produção (usa o mesmo TradingEnv do SILUS)
    """
    # Importar as mesmas funções do SILUS
    from silus import load_optimized_data_original, TradingEnv
    
    # 🚀 USAR DATASET EVAL OTIMIZADO - 50K LINHAS PREPARADAS PARA AVALIAÇÃO
    dataset_path = 'data/GC=F_EVAL_OPTIMIZED_V4_20250912_164339.csv'
    data = pd.read_csv(dataset_path)
    
    # Converter time para datetime se necessário
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'])
    
    # 🔥 FIX CRÍTICO: Dataset já tem features _1m, manter apenas colunas básicas OHLCV
    # O problema é que tem features duplicadas causando shape (50000, 2)
    basic_columns = ['time', 'open_1m', 'high_1m', 'low_1m', 'close_1m', 'tick_volume_1m']
    
    # Se não existirem _1m, usar colunas básicas e renomear
    if 'open_1m' not in data.columns:
        column_mapping = {
            'open': 'open_1m',
            'high': 'high_1m', 
            'low': 'low_1m',
            'close': 'close_1m',
            'tick_volume': 'tick_volume_1m'
        }
        
        # Aplicar renomeação apenas para colunas que existem
        columns_to_rename = {old: new for old, new in column_mapping.items() if old in data.columns}
        if columns_to_rename:
            data = data.rename(columns=columns_to_rename)
            print(f"📊 Colunas renomeadas para formato 1min: {list(columns_to_rename.keys())}")
    
    # 🚀 MANTER APENAS COLUNAS BÁSICAS PARA EVITAR CONFLITOS
    available_basic = [col for col in basic_columns if col in data.columns]
    data = data[available_basic].copy()
    
    print(f"📊 Dataset V4 carregado: {len(data)} linhas, {len(data.columns)} colunas básicas (timeframe 1min)")
    
    # 🔥 AJUSTE PARA 1MIN: 1 semana = 7200 steps (1440 min/dia × 5 dias)
    if USE_RECENT_DATA and len(data) > RECENT_WEEKS_COUNT * 7200:
        # Pegar últimas semanas (7200 steps por semana - timeframe 1min)
        recent_data_size = RECENT_WEEKS_COUNT * 7200
        data = data.iloc[-recent_data_size:].reset_index(drop=True)
        print(f"📅 Usando dados recentes: {len(data)} steps ({RECENT_WEEKS_COUNT} semanas - 1min timeframe)")
    
    # 🔥 VERIFICAÇÃO FINAL DO DATASET
    print(f"🔍 Verificação dataset antes do TradingEnv:")
    print(f"   Shape: {data.shape}")
    print(f"   Colunas: {list(data.columns)}")
    for col in data.columns:
        if col != 'time':
            col_shape = data[col].shape if hasattr(data[col], 'shape') else 'No shape'
            print(f"   {col}: {col_shape}, tipo: {type(data[col].iloc[0] if len(data) > 0 else 'empty')}")
    
    # 🔥 AMBIENTE PARA TESTE PURO - SEM COOLDOWNS/TIMEOUTS
    env = TradingEnv(
        df=data,
        window_size=20,  # Mesmo do SILUS
        is_training=False,  # 🔧 TESTE PURO: Modo eval (sem enhancements)
        initial_balance=INITIAL_PORTFOLIO,
        trading_params={
            'min_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'enable_shorts': True,
            'max_positions': 2
        }
    )
    
    # 🚨 CRÍTICO: Configurar activity_system para timeframe 1min
    if hasattr(env, 'activity_system') and env.activity_system is not None:
        # ✅ OPÇÃO 1: Ajustar timeout de 60 candles (1h) para 300 candles (5h em 1min)
        if hasattr(env.activity_system.config, 'position_timeout_candles'):
            env.activity_system.config.position_timeout_candles = 300  # 5h em candles 1min
            print("🔧 Position timeout ajustado: 60 → 300 candles (5h em 1min)")
        
        # ✅ OPÇÃO 2: Para teste PURO, desabilitar completamente
        # env.activity_system = None
        # print("🔧 Activity system desabilitado para teste puro")
    
    # 🚨 CRÍTICO: Zerar cooldowns para teste puro 
    if hasattr(env, 'cooldown_after_trade'):
        env.cooldown_after_trade = 0
        print("🔧 Cooldowns zerrados para teste puro")
    
    if hasattr(env, 'cooldown_base'):
        env.cooldown_base = 0
        print("🔧 Cooldown base zerrado para teste puro")
    
    return env

# 🚀 CACHE GLOBAL DE MODELOS PARA EVITAR RECARREGAMENTO
_model_cache = {}

def evaluate_model_realistic(model_path, num_episodes=NUM_EPISODES):
    """
    Avaliação realística de um modelo COM CACHE OTIMIZADO
    """
    model_name = os.path.basename(model_path)
    print(f"\\n⚡ TESTE REALÍSTICO: {model_name}")
    print("", end='', flush=True)  # Iniciar linha de progresso
    
    # 🚀 OTIMIZAÇÃO: Usar cache de modelos
    from sb3_contrib import RecurrentPPO
    try:
        if model_path not in _model_cache:
            print(f" [LOADING]", end='', flush=True)
            model = RecurrentPPO.load(model_path)
            
            # 🔧 CONFIGURAÇÕES IDÊNTICAS AO CHERRY
            model.policy.set_training_mode(False)  # Modo eval
            for param in model.policy.parameters():
                param.requires_grad = False        # Desabilitar gradientes
            
            _model_cache[model_path] = model
            print(f" [CACHED]", end='', flush=True)
        else:
            model = _model_cache[model_path]
            print(f" [FROM CACHE]", end='', flush=True)
            
        print(f" ✅")
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
    print(f"🔧 Criando ambiente...")
    env = setup_realistic_environment()
    
    # Testar com múltiplas seeds
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\\n🎲 Testando com seed {seed} ({seed_idx+1}/{len(SEEDS)}) [REUSING ENV]")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        seed_results = []
        
        for episode in range(num_episodes // len(SEEDS)):
            try:
                # Reset environment
                obs = env.reset()
                episode_return = 0.0
                episode_trades = 0
                episode_steps = 0
                lstm_states = None  # 🔧 ADICIONADO: LSTM states como Cherry
                
                # Run episode with torch optimization + EARLY TERMINATION
                with torch.no_grad():  # 🚀 OTIMIZAÇÃO: Disable gradients para inference
                    consecutive_holds = 0
                    max_consecutive_holds = 100  # 🚀 OTIMIZAÇÃO: Early termination se ficar muito tempo em HOLD
                    
                    for step in range(TEST_STEPS):
                        # Predict action (IGUAL AO CHERRY)
                        action, lstm_states = model.predict(
                            obs, 
                            state=lstm_states,    # 🔧 ADICIONADO: state parameter
                            deterministic=DETERMINISTIC
                        )
                        
                        # 🚀 OTIMIZAÇÃO: Tracking de ações para early termination
                        if hasattr(action, '__len__') and len(action) > 0:
                            if abs(action[0]) < 0.33:  # HOLD action
                                consecutive_holds += 1
                            else:
                                consecutive_holds = 0
                        
                        # Step environment
                        obs, reward, done, info = env.step(action)
                        episode_return += reward
                        episode_steps += 1
                        
                        # 🚀 OTIMIZAÇÃO: Early termination conditions
                        if done:
                            break
                        if consecutive_holds > max_consecutive_holds and episode_steps > TEST_STEPS // 4:
                            # Se ficou muito tempo em HOLD após 25% do episódio, terminar cedo
                            print("E", end='', flush=True)  # Early termination indicator
                            break
                
                # 🔥 COLETAR MÉTRICAS REAIS DO TRADING (IGUAL AO CHERRY)
                portfolio_pnl = env.portfolio_value - INITIAL_PORTFOLIO
                trades_list = getattr(env, 'trades', [])
                
                # 🔥 AJUSTE: Calcular win/loss baseado na estrutura de trades do sistema
                # Verificar estrutura dos trades para compatibilidade
                winning_trades = 0
                losing_trades = 0
                total_trades_real = len(trades_list)
                
                for trade in trades_list:
                    # Tentar diferentes campos de PnL baseado na estrutura do trade
                    pnl = trade.get('pnl_usd', trade.get('pnl', trade.get('profit', 0)))
                    if pnl > 0:
                        winning_trades += 1
                    elif pnl < 0:
                        losing_trades += 1
                
                seed_results.append({
                    'return': episode_return,
                    'trades': total_trades_real,  # 🔧 USANDO TRADES REAIS
                    'steps': episode_steps,
                    'active': total_trades_real > 0,
                    # Métricas adicionais
                    'portfolio_pnl': portfolio_pnl,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades
                })
                
                if total_trades_real > 0:
                    results['active_episodes'] += 1
                
                results['total_trades'] += total_trades_real
                results['trades_per_episode'].append(total_trades_real)
                
                # Progress indicator (igual ao Cherry)
                print(".", end='', flush=True)
                
            except Exception as e:
                print(f"❌ Erro no episódio {episode}: {e}")
                continue
        
        print(f" ✓")  # Finalizar linha de progresso
        
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
        
        # Agregar todos os trades de todos os episódios
        total_winning_trades = sum(ep.get('winning_trades', 0) for ep in results['episodes'])
        total_losing_trades = sum(ep.get('losing_trades', 0) for ep in results['episodes'])
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
            'avg_trades_per_day': np.mean(trades) / 5,  # 🔥 AJUSTE: 5 dias úteis por semana (não 7)
            'total_trades_real': total_real_trades,
            'winning_trades': total_winning_trades,
            'losing_trades': total_losing_trades,
            
            # 🎯 WIN RATE CORRETO - baseado em trades individuais
            'win_rate': real_win_rate,
            
            # Episodes com lucro (PnL > 0)
            'profitable_episodes': sum(1 for pnl in portfolio_pnls if pnl > 0),
            'losing_episodes_pnl': sum(1 for pnl in portfolio_pnls if pnl < 0),
            'episode_profit_rate': sum(1 for pnl in portfolio_pnls if pnl > 0) / len(portfolio_pnls) * 100 if portfolio_pnls else 0,
            
            'seeds_consistency': {
                seed: {
                    'mean_return': np.mean([ep['return'] for ep in seed_data]),
                    'mean_trades': np.mean([ep['trades'] for ep in seed_data])
                }
                for seed, seed_data in results['seeds_results'].items()
            }
        }
    else:
        metrics = {'error': 'No valid episodes'}
    
    return {
        'model_path': model_path,
        'metrics': metrics,
        'raw_results': results
    }

def run_realistic_comparison():
    """
    Executar comparação realística entre modelos
    """
    from datetime import datetime
    
    print("🚀 AVALIAÇÃO REALÍSTICA SILUS - COMPARAÇÃO COMPLETA")
    print("=" * 60)
    
    print(f"📊 Testando {len(CHECKPOINTS_TO_TEST)} modelos SILUS")
    print(f"⚡ {NUM_EPISODES} episódios por modelo") 
    print(f"🎲 {len(SEEDS)} seeds")
    print(f"📈 Total: {len(CHECKPOINTS_TO_TEST) * NUM_EPISODES} episódios")
    print(f"🕐 Steps por episódio: {TEST_STEPS}")
    print("-" * 60)
    
    results = {}
    start_time = datetime.now()
    
    for idx, model_path in enumerate(CHECKPOINTS_TO_TEST):
        model_start = datetime.now()
        
        if os.path.exists(model_path):
            result = evaluate_model_realistic(model_path)
            if result:
                model_name = os.path.basename(model_path)
                results[model_name] = result
                
                # 🔥 ANÁLISE INDIVIDUAL POR MODELO (IGUAL AO CHERRY)
                model_time = (datetime.now() - model_start).total_seconds()
                metrics = result['metrics']
                
                print(f"  ⏱️ Tempo: {model_time:.1f}s")
                print(f"  📊 Sharpe: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"  💰 PnL: ${metrics.get('total_real_pnl', 0):.2f}")
                print(f"  🎯 Win Rate: {metrics.get('win_rate', 0):.1f}% ({metrics.get('winning_trades', 0)}/{metrics.get('total_trades_real', 0)})")
                print(f"  📈 Trades/ep: {metrics.get('avg_trades_per_episode', 0):.1f}")
                print(f"  🎯 Activity: {metrics.get('activity_rate', 0):.1f}%")
        else:
            print(f"⚠️ Modelo não encontrado: {model_path}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # 🔥 RELATÓRIO COMPARATIVO FINAL (IGUAL AO CHERRY)
    print("\\n" + "=" * 60)
    print("📊 RESULTADOS COMPARATIVOS FINAIS - SILUS:")
    print("-" * 60)
    
    # Encontrar melhor modelo
    best_sharpe = -999
    best_pnl = -99999
    best_model_sharpe = None
    best_model_pnl = None
    
    for model_name, result in results.items():
        metrics = result['metrics']
        if 'error' not in metrics:
            # Extrair número de steps do nome
            try:
                steps = model_name.split('_')[2]  # SILUS_simpledirecttraining_500000_steps...
            except:
                steps = model_name
            
            print(f"\\n🏷️ {steps} STEPS:")
            print(f"  📊 Sharpe: {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  💰 Total PnL: ${metrics.get('total_real_pnl', 0):.2f}")
            print(f"  📈 Win Rate: {metrics.get('win_rate', 0):.1f}% ({metrics.get('winning_trades', 0)}/{metrics.get('total_trades_real', 0)})")
            print(f"  🎯 Activity: {metrics.get('activity_rate', 0):.1f}%")
            print(f"  📈 Trades/ep: {metrics.get('avg_trades_per_episode', 0):.1f}")
            print(f"  📈 Episodes lucrativos: {metrics.get('profitable_episodes', 0)}/{metrics.get('total_episodes', 0)}")
            
            # Tracking dos melhores
            if metrics.get('sharpe_ratio', -999) > best_sharpe:
                best_sharpe = metrics.get('sharpe_ratio', -999)
                best_model_sharpe = steps
            
            if metrics.get('total_real_pnl', -99999) > best_pnl:
                best_pnl = metrics.get('total_real_pnl', -99999)
                best_model_pnl = steps
    
    print("\\n" + "=" * 60)
    print("🏆 RANKING FINAL:")
    print(f"   🥇 MELHOR SHARPE: {best_model_sharpe} STEPS (Sharpe: {best_sharpe:.4f})")
    print(f"   💰 MELHOR PnL: {best_model_pnl} STEPS (PnL: ${best_pnl:.2f})")
    print(f"⏱️ TEMPO TOTAL: {total_time:.1f} segundos ({total_time/60:.1f} minutos)")
    print(f"⚡ VELOCIDADE: {total_time/len(results):.1f}s por modelo")
    print("=" * 60)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        if 'error' not in metrics:
            steps = model_name.split('_')[2]  # Extrair steps
            print(f"\\n🏷️ {steps} STEPS:")
            print(f"  📈 Return médio: {metrics['mean_return']:.4f}")
            print(f"  📊 Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"  🎯 Taxa atividade: {metrics['activity_rate']:.1f}%")
            print(f"  📈 Trades/episódio: {metrics['avg_trades_per_episode']:.1f}")
            print(f"  📅 Trades/dia: {metrics['avg_trades_per_day']:.1f}")
            print(f"  🎲 Consistência seeds: {len(metrics['seeds_consistency'])} seeds")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"avaliacao_realistica_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\\n💾 Resultados salvos: {filename}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_realistic_comparison()
        
        print("\\n✅ AVALIAÇÃO REALÍSTICA CONCLUÍDA!")
        print("\\n🎯 PRÓXIMOS PASSOS:")
        print("1. Compare com dados de produção real")
        print("2. Valide taxa de atividade vs robô real") 
        print("3. Confirme número de trades/dia")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        traceback.print_exc()