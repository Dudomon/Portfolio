#!/usr/bin/env python3
"""
🚀 AVALIAÇÃO REALÍSTICA V11 - REPLICAR COMPORTAMENTO DE PRODUÇÃO
================================================================

CORREÇÕES PARA TESTE CONFIÁVEL:
✅ 1. Dados sequenciais recentes (não aleatórios)
✅ 2. Configuração idêntica à produção
✅ 3. Períodos completos (1 semana = 2016 steps)
✅ 4. Stochastic mode (exploration ativa)
✅ 5. Métricas realistas de trading
✅ 6. Multiple seeds para robustez
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

# CONFIGURAÇÃO REALÍSTICA - REPLICAR PRODUÇÃO
CHECKPOINTS_TO_TEST = [
    "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_500000_steps_20250908_204556.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_750000_steps_20250908_210824.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_1000000_steps_20250908_213057.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_1250000_steps_20250908_215346.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_1500000_steps_20250908_221640.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_1750000_steps_20250908_223938.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_2000000_steps_20250908_230256.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_2250000_steps_20250908_232550.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_2500000_steps_20250908_234851.zip"
]

# PARÂMETROS REALÍSTICOS
INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03

# TESTE REALÍSTICO - REPLICAR PRODUÇÃO (IGUAL AO CHERRY)
TEST_STEPS = 1800          # 🔧 CORRIGIDO: Same as Cherry (1 semana de trading 5m bars)
NUM_EPISODES = 25          # 🔧 CORRIGIDO: Same as Cherry (25 episódios)
SEEDS = [42, 123]          # 🔧 CORRIGIDO: Same as Cherry (2 seeds)
DETERMINISTIC = False      # Non-deterministic para avaliação realística
CONFIDENCE_THRESHOLD = 0.3 # Baixo como produção

# USAR DADOS SEQUENCIAIS RECENTES
USE_RECENT_DATA = True
RECENT_WEEKS_COUNT = 30    # Últimas 30 semanas

def setup_realistic_environment():
    """
    Configurar ambiente IDÊNTICO à produção (usa o mesmo TradingEnv do SILUS)
    """
    # Importar as mesmas funções do SILUS
    from silus import load_optimized_data_original, TradingEnv
    
    # Dados iguais aos do SILUS
    data = load_optimized_data_original()
    
    # Usar dados recentes se solicitado
    if USE_RECENT_DATA and len(data) > RECENT_WEEKS_COUNT * 2016:
        # Pegar últimas semanas (2016 steps por semana)
        recent_data_size = RECENT_WEEKS_COUNT * 2016
        data = data.iloc[-recent_data_size:].reset_index(drop=True)
        print(f"📅 Usando dados recentes: {len(data)} steps ({RECENT_WEEKS_COUNT} semanas)")
    
    # Ambiente IDÊNTICO ao SILUS
    env = TradingEnv(
        df=data,
        window_size=20,  # Mesmo do SILUS
        is_training=True,  # 🔧 CORRIGIDO: Mesmo modo que Cherry (affects trading behavior)
        initial_balance=INITIAL_PORTFOLIO,
        trading_params={
            'min_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'enable_shorts': True,
            'max_positions': 2
        }
    )
    
    return env

def evaluate_model_realistic(model_path, num_episodes=NUM_EPISODES):
    """
    Avaliação realística de um modelo
    """
    model_name = os.path.basename(model_path)
    print(f"\\n⚡ TESTE REALÍSTICO: {model_name}")
    print("", end='', flush=True)  # Iniciar linha de progresso
    
    # Carregar modelo 
    from sb3_contrib import RecurrentPPO
    try:
        model = RecurrentPPO.load(model_path)
        
        # 🔧 CONFIGURAÇÕES IDÊNTICAS AO CHERRY
        model.policy.set_training_mode(False)  # Modo eval
        for param in model.policy.parameters():
            param.requires_grad = False        # Desabilitar gradientes
            
        print(f"✅ Modelo carregado: {os.path.basename(model_path)}")
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
    
    # Testar com múltiplas seeds
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\\n🎲 Testando com seed {seed} ({seed_idx+1}/{len(SEEDS)})")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        seed_results = []
        
        # Setup environment
        env = setup_realistic_environment()
        
        for episode in range(num_episodes // len(SEEDS)):
            try:
                # Reset environment
                obs = env.reset()
                episode_return = 0.0
                episode_trades = 0
                episode_steps = 0
                lstm_states = None  # 🔧 ADICIONADO: LSTM states como Cherry
                
                # Run episode with torch optimization (IGUAL AO CHERRY)
                with torch.no_grad():  # 🚀 OTIMIZAÇÃO: Disable gradients para inference
                    for step in range(TEST_STEPS):
                        # Predict action (IGUAL AO CHERRY)
                        action, lstm_states = model.predict(
                            obs, 
                            state=lstm_states,    # 🔧 ADICIONADO: state parameter
                            deterministic=DETERMINISTIC
                        )
                        
                        # Step environment
                        obs, reward, done, info = env.step(action)
                        episode_return += reward
                        episode_steps += 1
                        
                        if done:
                            break
                
                # 🔥 COLETAR MÉTRICAS REAIS DO TRADING (IGUAL AO CHERRY)
                portfolio_pnl = env.portfolio_value - INITIAL_PORTFOLIO
                trades_list = getattr(env, 'trades', [])
                
                # Calcular win/loss de trades individuais
                winning_trades = sum(1 for trade in trades_list if trade.get('pnl_usd', 0) > 0)
                losing_trades = sum(1 for trade in trades_list if trade.get('pnl_usd', 0) < 0)
                total_trades_real = len(trades_list)
                
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
            'avg_trades_per_day': np.mean(trades) / 7,
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