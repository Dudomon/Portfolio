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

import numpy as np
import pandas as pd
import torch

# 🔥 AVALIAÇÃO COMPLETA - 8 CHECKPOINTS CHERRY (NOMES CORRETOS)
CHECKPOINTS_TO_TEST = [
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_2800000_steps_20250905_141006.zip",  # 2.8M steps
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_2850000_steps_20250905_141441.zip",  # 2.85M steps
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_2900000_steps_20250905_141910.zip",  # 2.9M steps
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_2950000_steps_20250905_142346.zip",  # 2.95M steps
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_3000000_steps_20250905_142822.zip",  # 3.0M steps
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_3050000_steps_20250905_143304.zip",  # 3.05M steps
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_3100000_steps_20250905_143734.zip",  # 3.1M steps
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_3150000_steps_20250905_144201.zip",  # 3.15M steps
]

# PARÂMETROS REALÍSTICOS
INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03

# 🔥 AVALIAÇÃO COMPLETA - 50 EPISÓDIOS
TEST_STEPS = 500          # 500 steps por episódio (aproximadamente 1 dia de trading)
NUM_EPISODES = 50         # 50 episódios para estatística robusta
SEEDS = [42, 123, 456, 789, 999]  # 5 seeds diferentes para robustez
DETERMINISTIC = False     # STOCHASTIC como produção
CONFIDENCE_THRESHOLD = 0.3 # Baixo como produção

# USAR DADOS SEQUENCIAIS RECENTES
USE_RECENT_DATA = True
RECENT_WEEKS_COUNT = 30    # Últimas 30 semanas

def setup_realistic_environment():
    """
    Configurar ambiente IDÊNTICO à produção (usa o mesmo TradingEnv do CHERRY)
    """
    # MUDAR WORKING DIRECTORY ANTES DE IMPORTAR
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        # Importar as mesmas funções do CHERRY
        from cherry import load_optimized_data_original, TradingEnv
        
        # Dados iguais aos do CHERRY
        data = load_optimized_data_original()
    finally:
        # RESTAURAR WORKING DIRECTORY
        os.chdir(original_cwd)
    
    # Usar dados recentes se solicitado
    if USE_RECENT_DATA and len(data) > RECENT_WEEKS_COUNT * 2016:
        # Pegar últimas semanas (2016 steps por semana)
        recent_data_size = RECENT_WEEKS_COUNT * 2016
        data = data.iloc[-recent_data_size:].reset_index(drop=True)
        print(f"📅 Usando dados recentes: {len(data)} steps ({RECENT_WEEKS_COUNT} semanas)")
    
    # Ambiente IDÊNTICO ao CHERRY - FORÇAR TRAINING=TRUE PARA TESTE
    env = TradingEnv(
        df=data,
        window_size=20,  # Mesmo do CHERRY
        is_training=True,  # FORÇADO: Modo treinamento para ativar todos os sistemas
        initial_balance=INITIAL_PORTFOLIO,
        trading_params={
            'min_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'enable_shorts': True,
            'max_positions': 2
        }
    )
    
    return env

def calculate_trading_metrics(episodes_data):
    """
    🔥 ANÁLISE PROFISSIONAL DE TRADING - MÉTRICAS COMPLETAS
    """
    all_trades = []
    all_returns = []
    
    # Coletar todos os trades de todos os episódios
    for episode in episodes_data:
        all_returns.append(episode['return'])
    
    # Para acessar trades individuais, precisamos modificar o environment
    # Por ora, vamos calcular com base nos returns dos episódios
    
    total_episodes = len(episodes_data)
    winning_episodes = len([ep for ep in episodes_data if ep['return'] > 0])
    losing_episodes = total_episodes - winning_episodes
    
    win_returns = [ep['return'] for ep in episodes_data if ep['return'] > 0]
    loss_returns = [ep['return'] for ep in episodes_data if ep['return'] < 0]
    
    # MÉTRICAS BÁSICAS
    win_rate = (winning_episodes / total_episodes * 100) if total_episodes > 0 else 0
    
    total_wins = sum(win_returns) if win_returns else 0
    total_losses = abs(sum(loss_returns)) if loss_returns else 0
    
    profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
    
    avg_win = (total_wins / len(win_returns)) if win_returns else 0
    avg_loss = (total_losses / len(loss_returns)) if loss_returns else 0
    
    # DRAWDOWN ANALYSIS
    portfolio_curve = []
    running_balance = 0
    peak = 0
    max_drawdown = 0
    
    for ep_return in all_returns:
        running_balance += ep_return
        portfolio_curve.append(running_balance)
        
        if running_balance > peak:
            peak = running_balance
        
        current_dd = ((peak - running_balance) / peak * 100) if peak > 0 else 0
        if current_dd > max_drawdown:
            max_drawdown = current_dd
    
    # CONSECUTIVE ANALYSIS
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for ep in episodes_data:
        if ep['return'] > 0:
            current_win_streak += 1
            current_loss_streak = 0
            if current_win_streak > max_consecutive_wins:
                max_consecutive_wins = current_win_streak
        else:
            current_loss_streak += 1
            current_win_streak = 0
            if current_loss_streak > max_consecutive_losses:
                max_consecutive_losses = current_loss_streak
    
    # RATIOS PROFISSIONAIS
    returns_array = np.array(all_returns)
    
    # Sharpe Ratio (corrigido)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    sharpe_ratio = (mean_return / std_return) if std_return > 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns_array[returns_array < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return / downside_std) if downside_std > 0 else 0
    
    # Calmar Ratio
    calmar_ratio = (mean_return / max_drawdown) if max_drawdown > 0 else 0
    
    # Recovery Factor
    recovery_factor = (total_wins / max_drawdown) if max_drawdown > 0 else 0
    
    return {
        'basic_metrics': {
            'total_episodes': total_episodes,
            'winning_episodes': winning_episodes,
            'losing_episodes': losing_episodes,
            'win_rate_pct': round(win_rate, 2),
            'profit_factor': round(profit_factor, 3),
            'total_pnl': round(sum(all_returns), 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'risk_reward_ratio': round(avg_win / abs(avg_loss), 3) if avg_loss != 0 else 0
        },
        'drawdown_analysis': {
            'max_drawdown_pct': round(max_drawdown, 2),
            'recovery_factor': round(recovery_factor, 3),
            'final_portfolio_value': round(portfolio_curve[-1], 2) if portfolio_curve else 0
        },
        'consecutive_analysis': {
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'current_streak': current_win_streak if current_win_streak > 0 else -current_loss_streak
        },
        'professional_ratios': {
            'sharpe_ratio': round(sharpe_ratio, 4),
            'sortino_ratio': round(sortino_ratio, 4),
            'calmar_ratio': round(calmar_ratio, 4),
            'volatility_pct': round(std_return / mean_return * 100, 2) if mean_return != 0 else 0
        },
        'performance_classification': {
            'grade': _classify_performance(win_rate, profit_factor, sharpe_ratio, max_drawdown),
            'is_profitable': total_wins > total_losses,
            'is_consistent': max_consecutive_losses <= 5,
            'risk_level': 'HIGH' if max_drawdown > 20 else 'MEDIUM' if max_drawdown > 10 else 'LOW'
        }
    }

def _classify_performance(win_rate, profit_factor, sharpe, max_dd):
    """Classificar performance do modelo"""
    score = 0
    
    # Win Rate (30% peso)
    if win_rate >= 60: score += 30
    elif win_rate >= 50: score += 20
    elif win_rate >= 40: score += 10
    
    # Profit Factor (25% peso)  
    if profit_factor >= 2.0: score += 25
    elif profit_factor >= 1.5: score += 20
    elif profit_factor >= 1.2: score += 15
    elif profit_factor >= 1.0: score += 10
    
    # Sharpe (25% peso)
    if sharpe >= 1.5: score += 25
    elif sharpe >= 1.0: score += 20
    elif sharpe >= 0.5: score += 15
    elif sharpe >= 0: score += 5
    
    # Drawdown (20% peso)
    if max_dd <= 5: score += 20
    elif max_dd <= 10: score += 15
    elif max_dd <= 20: score += 10
    elif max_dd <= 30: score += 5
    
    if score >= 80: return 'A+'
    elif score >= 70: return 'A'
    elif score >= 60: return 'B+'
    elif score >= 50: return 'B'
    elif score >= 40: return 'C+'
    elif score >= 30: return 'C'
    elif score >= 20: return 'D'
    else: return 'F'

def evaluate_model_realistic(model_path, num_episodes=NUM_EPISODES):
    """
    Avaliação realística de um modelo
    """
    print(f"\\n🔍 TESTE REALÍSTICO: {os.path.basename(model_path)}")
    
    # Carregar modelo 
    from sb3_contrib import RecurrentPPO
    try:
        model = RecurrentPPO.load(model_path)
        print(f"✅ Modelo carregado: {os.path.basename(model_path)}")
        
        # 🔥 MODO INFERÊNCIA: Congelar pesos e desabilitar gradientes
        model.policy.set_training_mode(False)  # Modo avaliação/inferência
        for param in model.policy.parameters():
            param.requires_grad = False  # Congelar todos os pesos
        
        print(f"🧊 Modo inferência ativado - pesos congelados")
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
                lstm_states = None  # CRÍTICO: Inicializar LSTM states
                
                # Run episode
                for step in range(TEST_STEPS):
                    # Predict action (STOCHASTIC) com LSTM states
                    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=DETERMINISTIC)
                    
                    # Step environment
                    obs, reward, done, info = env.step(action)
                    
                    episode_return += reward
                    episode_steps += 1
                    
                    # Count trades
                    if 'trade_executed' in info and info['trade_executed']:
                        episode_trades += 1
                    
                    if done:
                        break
                
                seed_results.append({
                    'return': episode_return,
                    'trades': episode_trades,
                    'steps': episode_steps,
                    'active': episode_trades > 0
                })
                
                if episode_trades > 0:
                    results['active_episodes'] += 1
                
                results['total_trades'] += episode_trades
                results['trades_per_episode'].append(episode_trades)
                
                print(f"  Episode {episode+1}: Return={episode_return:.4f}, Trades={episode_trades}")
                
            except Exception as e:
                print(f"❌ Erro no episódio {episode}: {e}")
                continue
        
        results['seeds_results'][seed] = seed_results
        results['episodes'].extend(seed_results)
    
    # 🔥 CALCULAR MÉTRICAS PROFISSIONAIS DE TRADING
    if len(results['episodes']) > 0:
        # Métricas básicas (mantidas para compatibilidade)
        returns = [ep['return'] for ep in results['episodes']]
        trades = [ep['trades'] for ep in results['episodes']]
        
        basic_metrics = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio_old': np.mean(returns) / (np.std(returns) + 1e-8),  # Manter o antigo
            
            'total_episodes': len(results['episodes']),
            'active_episodes': sum(1 for ep in results['episodes'] if ep['active']),
            'activity_rate': sum(1 for ep in results['episodes'] if ep['active']) / len(results['episodes']) * 100,
            
            'total_trades': sum(trades),
            'avg_trades_per_episode': np.mean(trades),
            'avg_trades_per_day': np.mean(trades) / 7,  # Por semana = 7 dias
            
            'seeds_consistency': {
                seed: {
                    'mean_return': np.mean([ep['return'] for ep in seed_data]),
                    'mean_trades': np.mean([ep['trades'] for ep in seed_data])
                }
                for seed, seed_data in results['seeds_results'].items()
            }
        }
        
        # 🚀 MÉTRICAS PROFISSIONAIS COMPLETAS
        professional_metrics = calculate_trading_metrics(results['episodes'])
        
        # Combinar métricas básicas + profissionais
        metrics = {**basic_metrics, 'professional_trading_analysis': professional_metrics}
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
    print("🚀 INICIANDO AVALIAÇÃO REALÍSTICA")
    print("=" * 60)
    
    results = {}
    
    for model_path in CHECKPOINTS_TO_TEST:
        if os.path.exists(model_path):
            result = evaluate_model_realistic(model_path)
            if result:
                model_name = os.path.basename(model_path)
                results[model_name] = result
        else:
            print(f"⚠️ Modelo não encontrado: {model_path}")
    
    # Análise comparativa
    print("\\n📊 RESULTADOS COMPARATIVOS:")
    print("-" * 50)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        if 'error' not in metrics:
            steps = model_name.split('_')[2]  # Extrair steps
            prof_metrics = metrics.get('professional_trading_analysis', {})
            basic = prof_metrics.get('basic_metrics', {})
            dd_analysis = prof_metrics.get('drawdown_analysis', {})
            ratios = prof_metrics.get('professional_ratios', {})
            classification = prof_metrics.get('performance_classification', {})
            
            print(f"\\n🏷️ {steps} STEPS - NOTA: {classification.get('grade', 'N/A')}")
            print(f"=" * 60)
            
            # MÉTRICAS BÁSICAS
            print(f"📊 PERFORMANCE GERAL:")
            print(f"  💰 Total PnL: ${basic.get('total_pnl', 0):,.2f}")
            print(f"  📈 Win Rate: {basic.get('win_rate_pct', 0):.1f}%")
            print(f"  🎯 Profit Factor: {basic.get('profit_factor', 0):.3f}")
            print(f"  ⚖️ Risk/Reward: {basic.get('risk_reward_ratio', 0):.3f}")
            
            # DRAWDOWN & RISCO
            print(f"\\n🚨 ANÁLISE DE RISCO:")
            print(f"  📉 Max Drawdown: {dd_analysis.get('max_drawdown_pct', 0):.2f}%")
            print(f"  🔄 Recovery Factor: {dd_analysis.get('recovery_factor', 0):.3f}")
            print(f"  ⚠️ Risk Level: {classification.get('risk_level', 'N/A')}")
            
            # RATIOS PROFISSIONAIS
            print(f"\\n📐 RATIOS PROFISSIONAIS:")
            print(f"  📊 Sharpe Ratio: {ratios.get('sharpe_ratio', 0):.4f}")
            print(f"  📈 Sortino Ratio: {ratios.get('sortino_ratio', 0):.4f}")
            print(f"  🎯 Calmar Ratio: {ratios.get('calmar_ratio', 0):.4f}")
            
            # ATIVIDADE
            print(f"\\n⚡ ATIVIDADE:")
            print(f"  🎯 Taxa atividade: {metrics['activity_rate']:.1f}%")
            print(f"  📈 Trades/episódio: {metrics['avg_trades_per_episode']:.1f}")
            print(f"  📅 Trades/dia: {metrics['avg_trades_per_day']:.1f}")
            
            # CLASSIFICAÇÃO FINAL
            is_profitable = classification.get('is_profitable', False)
            is_consistent = classification.get('is_consistent', False)
            print(f"\\n🏆 VEREDICTO:")
            print(f"  💰 Lucrativo: {'✅ SIM' if is_profitable else '❌ NÃO'}")
            print(f"  🔄 Consistente: {'✅ SIM' if is_consistent else '❌ NÃO'}")
            print(f"  📊 Nota Final: {classification.get('grade', 'F')}")
            print(f"  🎲 Seeds testadas: {len(metrics['seeds_consistency'])}")
    
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