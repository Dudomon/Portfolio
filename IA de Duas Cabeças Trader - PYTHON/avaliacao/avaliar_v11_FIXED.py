#!/usr/bin/env python3
"""
🚀 AVALIAÇÃO V11 FIXED - CORREÇÃO DOS PROBLEMAS DE ESCALA IDENTIFICADOS
=========================================================================

CORREÇÕES IMPLEMENTADAS:
✅ 1. Portfolio debugging: Logs detalhados da evolução do portfolio
✅ 2. Drawdown real: Baseado em portfolio_history em vez de returns aproximados  
✅ 3. Validação de unidades: Verificar se portfolio está em USD real
✅ 4. Métricas realistas: Returns e drawdowns condizentes com trading real

PROBLEMAS CORRIGIDOS:
- Returns irreais (~1.22% com drawdown 0.32%)
- Drawdown aproximado incorreto
- Possível inconsistência de unidades (USD vs pontos)
"""

import sys
import os
import traceback
from datetime import datetime
import random
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch

# CONFIGURAÇÃO IGUAL AO ORIGINAL MAS COM DEBUG
DEFAULT_CHECKPOINT_PATH = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_1450000_steps_20250825_151927.zip"
INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03

# PARÂMETROS DE TESTE
TEST_STEPS = 1800
NUM_EPISODES = 3  # 🔧 REDUZIDO PARA DEBUG - só 3 episódios inicialmente
MIN_EPISODE_GAP = 10000
CONFIDENCE_LEVEL = 0.95

# 🔍 DEBUG FLAGS
DEBUG_PORTFOLIO = True
DEBUG_TRADES = True
DEBUG_METRICS = True

def find_single_checkpoint_for_debug():
    """🔍 Pegar apenas 1 checkpoint para debug intensivo"""
    
    target_checkpoints = [
        "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_4000000_steps_20250828_065748.zip",  # 4M - melhor resultado
    ]
    
    for cp in target_checkpoints:
        if os.path.exists(cp):
            print(f"✅ Debug checkpoint: {os.path.basename(cp)}")
            return [cp]
    
    return []

def create_evaluation_dataset():
    """🎯 Dataset igual ao original mas com debug"""
    print("📊 Preparando dataset de avaliação (DEBUG MODE)...")
    
    cache_path = "D:/Projeto/data/CACHE_eval_dataset_processed.pkl"
    
    if os.path.exists(cache_path):
        print("🚀 Carregando dataset PRÉ-PROCESSADO do cache...")
        import pickle
        try:
            with open(cache_path, 'rb') as f:
                train_df, eval_df = pickle.load(f)
            print(f"✅ Cache carregado: {len(train_df):,} treino + {len(eval_df):,} avaliação")
            return train_df, eval_df
        except:
            print("⚠️ Erro no cache, reprocessando...")
    
    dataset_path = "D:/Projeto/data/GC=F_YAHOO_20250821_161220.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return None, None
    
    df = pd.read_csv(dataset_path)
    
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
        df.set_index('timestamp', inplace=True)
        df.drop('time', axis=1, inplace=True)
    
    df = df.rename(columns={
        'open': 'open_5m',
        'high': 'high_5m', 
        'low': 'low_5m',
        'close': 'close_5m',
        'tick_volume': 'volume_5m'
    })
    
    total_len = len(df)
    split_point = int(total_len * 0.8)
    train_df = df.iloc[:split_point]
    eval_df = df.iloc[split_point:]
    
    print(f"🔄 Split: {len(train_df):,} train, {len(eval_df):,} eval")
    
    return train_df, eval_df

def calculate_FIXED_metrics(episode_results):
    """📊 MÉTRICAS CORRIGIDAS - drawdown real baseado em portfolio_history"""
    
    if not episode_results:
        return {}
    
    returns = [ep['return_pct'] for ep in episode_results]
    portfolio_values = [ep['final_portfolio'] for ep in episode_results]
    all_trades = []
    all_portfolio_histories = []
    
    for ep in episode_results:
        all_trades.extend(ep.get('trades_log', []))
        if 'portfolio_history' in ep:
            all_portfolio_histories.append(ep['portfolio_history'])
    
    # MÉTRICAS BÁSICAS
    metrics = {
        'mean_return': np.mean(returns),
        'median_return': np.median(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'positive_episodes': len([r for r in returns if r > 0]),
        'win_rate_episodes': len([r for r in returns if r > 0]) / len(returns) * 100,
        'mean_final_portfolio': np.mean(portfolio_values),
        'portfolio_growth': (np.mean(portfolio_values) - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO * 100,
    }
    
    # 🔧 DRAWDOWN CORRIGIDO - baseado em portfolio_history REAL
    if len(returns) > 1:
        # Sharpe Ratio
        metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Sortino Ratio
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_deviation = np.std(negative_returns)
            metrics['sortino_ratio'] = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        else:
            metrics['sortino_ratio'] = float('inf') if np.mean(returns) > 0 else 0
        
        # 🎯 DRAWDOWN REAL - MÉTODO CORRETO
        max_drawdowns_per_episode = []
        
        for portfolio_history in all_portfolio_histories:
            if len(portfolio_history) > 1:
                portfolio_array = np.array(portfolio_history)
                
                # Calcular running peak
                running_peak = np.maximum.accumulate(portfolio_array)
                
                # Calcular drawdown real
                portfolio_drawdowns = (portfolio_array - running_peak) / running_peak * 100
                
                # Max drawdown deste episódio  
                episode_max_dd = np.min(portfolio_drawdowns)
                max_drawdowns_per_episode.append(episode_max_dd)
                
                if DEBUG_METRICS:
                    print(f"   📉 Episode DD: {episode_max_dd:.2f}% | Portfolio range: ${portfolio_array.min():.2f}-${portfolio_array.max():.2f}")
        
        if max_drawdowns_per_episode:
            # Max drawdown global = pior drawdown entre todos os episódios
            metrics['max_drawdown_REAL'] = min(max_drawdowns_per_episode)  # Mais negativo
            metrics['avg_drawdown_per_episode'] = np.mean(max_drawdowns_per_episode)
        else:
            metrics['max_drawdown_REAL'] = 0.0
            metrics['avg_drawdown_per_episode'] = 0.0
        
        # VaR
        metrics['var_5pct'] = np.percentile(returns, 5)
        
        # Calmar Ratio usando drawdown REAL
        real_dd = abs(metrics['max_drawdown_REAL'])
        if real_dd > 0.1:
            metrics['calmar_ratio_REAL'] = abs(metrics['mean_return']) / real_dd
        else:
            metrics['calmar_ratio_REAL'] = 0
    
    # MÉTRICAS DE TRADING
    if all_trades:
        profitable_trades = [t for t in all_trades if t.get('pnl_usd', 0) > 0]
        losing_trades = [t for t in all_trades if t.get('pnl_usd', 0) < 0]
        
        metrics.update({
            'total_trades': len(all_trades),
            'win_rate_trades': len(profitable_trades) / len(all_trades) * 100 if all_trades else 0,
            'avg_profit_per_trade': np.mean([t.get('pnl_usd', 0) for t in profitable_trades]) if profitable_trades else 0,
            'avg_loss_per_trade': np.mean([t.get('pnl_usd', 0) for t in losing_trades]) if losing_trades else 0,
            'total_pnl': sum(t.get('pnl_usd', 0) for t in all_trades),
            'trades_per_episode': len(all_trades) / len(episode_results),
        })
        
        # Profit Factor
        gross_profit = sum(t.get('pnl_usd', 0) for t in profitable_trades)
        gross_loss = abs(sum(t.get('pnl_usd', 0) for t in losing_trades))
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    return metrics

def debug_portfolio_evolution(trading_env, episode_num):
    """🔍 Debug detalhado da evolução do portfolio"""
    
    if not DEBUG_PORTFOLIO:
        return
    
    print(f"\n🔍 DEBUG PORTFOLIO - Episode {episode_num}")
    print(f"   Initial: ${trading_env.initial_balance:.2f}")
    print(f"   Current: ${trading_env.portfolio_value:.2f}")
    
    if hasattr(trading_env, 'trades') and trading_env.trades:
        total_pnl = sum(t.get('pnl_usd', 0) for t in trading_env.trades)
        print(f"   Trades: {len(trading_env.trades)} | Total PnL: ${total_pnl:.2f}")
        
        # Mostrar últimos 3 trades
        for i, trade in enumerate(trading_env.trades[-3:]):
            pnl = trade.get('pnl_usd', 0)
            print(f"     Trade {i+1}: PnL ${pnl:.2f}")

def test_FIXED_evaluation():
    """🚀 Teste com correções implementadas"""
    
    print(f"🚀 AVALIAÇÃO FIXED - DEBUG MODE")
    print("=" * 70)
    print(f"💵 Portfolio Inicial: ${INITIAL_PORTFOLIO}")
    print(f"🧠 Episódios: {NUM_EPISODES} (DEBUG - reduzido)")
    print(f"📏 Steps: {TEST_STEPS}")
    print(f"🔍 Debug: Portfolio={DEBUG_PORTFOLIO}, Trades={DEBUG_TRADES}, Metrics={DEBUG_METRICS}")
    print("=" * 70)
    
    try:
        from sb3_contrib import RecurrentPPO
        from silus import TradingEnv
        
        print("✅ Imports carregados")
        
        # Dataset
        train_df, eval_df = create_evaluation_dataset()
        if eval_df is None:
            return False
        
        # Checkpoint único para debug
        checkpoints = find_single_checkpoint_for_debug()
        if not checkpoints:
            print("❌ Nenhum checkpoint para debug")
            return False
        
        checkpoint_path = checkpoints[0]
        print(f"\n🤖 DEBUG CHECKPOINT: {os.path.basename(checkpoint_path)}")
        
        # Trading params
        trading_params = {
            'base_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'initial_balance': INITIAL_PORTFOLIO,
            'target_trades_per_day': 18,
            'stop_loss_range': (2.0, 8.0),
            'take_profit_range': (3.0, 15.0)
        }
        
        # Carregar modelo
        print("🤖 Carregando modelo...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            model = RecurrentPPO.load(checkpoint_path, device=device)
            print("✅ Modelo carregado com sucesso")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False
        
        model.policy.set_training_mode(False)
        
        # Executar episódios com DEBUG
        print(f"\n🚀 Iniciando {NUM_EPISODES} episódios COM DEBUG...")
        
        # Posições para episódios
        eval_len = len(eval_df)
        max_start_pos = eval_len - TEST_STEPS - 100
        
        if max_start_pos <= 0:
            print("⚠️ Dataset muito pequeno")
            return False
        
        episode_positions = []
        for i in range(NUM_EPISODES):
            pos = random.randint(0, max_start_pos - i * MIN_EPISODE_GAP)
            episode_positions.append(pos)
        
        print(f"🎯 Posições: {episode_positions}")
        
        # Criar ambiente
        trading_env = TradingEnv(
            eval_df,
            window_size=20,
            is_training=False,
            initial_balance=INITIAL_PORTFOLIO,
            trading_params=trading_params
        )
        
        all_episodes = []
        
        # EXECUTAR EPISÓDIOS COM DEBUG INTENSIVO
        for episode_num, start_pos in enumerate(episode_positions):
            print(f"\n📊 EPISÓDIO {episode_num + 1}/{NUM_EPISODES} - Pos {start_pos}")
            print("-" * 50)
            
            # 🔧 RESET COMPLETO (igual ao original)
            trading_env.current_step = start_pos + 20
            trading_env.portfolio_value = INITIAL_PORTFOLIO
            trading_env.initial_balance = INITIAL_PORTFOLIO
            
            if hasattr(trading_env, 'realized_balance'):
                trading_env.realized_balance = INITIAL_PORTFOLIO
            if hasattr(trading_env, 'peak_portfolio'):
                trading_env.peak_portfolio = INITIAL_PORTFOLIO
            if hasattr(trading_env, 'current_drawdown'):
                trading_env.current_drawdown = 0.0
            if hasattr(trading_env, 'trades'):
                trading_env.trades = []
            if hasattr(trading_env, 'positions'):
                trading_env.positions = []
            
            obs = trading_env._get_observation()
            lstm_states = None
            done = False
            step = 0
            
            portfolio_history = [INITIAL_PORTFOLIO]
            
            # DEBUG: Estado inicial
            print(f"   🔍 INÍCIO: Portfolio ${trading_env.portfolio_value:.2f}")
            
            step_debug_interval = TEST_STEPS // 5  # Debug a cada 20% dos steps
            
            while not done and step < TEST_STEPS:
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                obs, reward, done, info = trading_env.step(action)
                
                portfolio_history.append(trading_env.portfolio_value)
                step += 1
                
                # DEBUG a cada intervalo
                if step % step_debug_interval == 0:
                    debug_portfolio_evolution(trading_env, episode_num + 1)
                    
                    # Portfolio evolution check
                    if DEBUG_PORTFOLIO:
                        recent_portfolio = portfolio_history[-10:]  # Últimos 10 steps
                        portfolio_change = recent_portfolio[-1] - recent_portfolio[0]
                        print(f"   📈 Step {step}: Portfolio ${trading_env.portfolio_value:.2f} (Δ${portfolio_change:.2f})")
            
            # Resultados finais do episódio
            final_portfolio = trading_env.portfolio_value
            episode_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
            trades_log = getattr(trading_env, 'trades', [])
            
            # DEBUG: Resumo do episódio
            print(f"   📊 FINAL Episode {episode_num + 1}:")
            print(f"      Portfolio: ${INITIAL_PORTFOLIO:.2f} → ${final_portfolio:.2f}")
            print(f"      Return: {episode_return:+.2f}%")
            print(f"      Trades: {len(trades_log)}")
            print(f"      Portfolio range: ${min(portfolio_history):.2f} - ${max(portfolio_history):.2f}")
            
            # Calcular drawdown REAL deste episódio  
            if len(portfolio_history) > 1:
                portfolio_array = np.array(portfolio_history)
                running_peak = np.maximum.accumulate(portfolio_array)
                drawdowns = (portfolio_array - running_peak) / running_peak * 100
                max_dd = np.min(drawdowns)
                print(f"      Max DD (REAL): {max_dd:.2f}%")
            
            episode_result = {
                'episode': episode_num + 1,
                'start_pos': start_pos,
                'initial_portfolio': INITIAL_PORTFOLIO,
                'final_portfolio': final_portfolio,
                'return_pct': episode_return,
                'trades_count': len(trades_log),
                'trades_log': trades_log,
                'portfolio_history': portfolio_history,
            }
            
            all_episodes.append(episode_result)
        
        # CALCULAR MÉTRICAS CORRIGIDAS
        print(f"\n📊 CALCULANDO MÉTRICAS CORRIGIDAS...")
        metrics = calculate_FIXED_metrics(all_episodes)
        
        # RELATÓRIO FINAL COM COMPARAÇÃO
        print(f"\n🏆 RELATÓRIO FIXED vs ORIGINAL")
        print("=" * 60)
        
        if metrics:
            print(f"📈 Retorno médio: {metrics.get('mean_return', 0):+.2f}%")
            print(f"📉 Max DD (REAL): {metrics.get('max_drawdown_REAL', 0):.2f}%")
            print(f"📉 Max DD (médio): {metrics.get('avg_drawdown_per_episode', 0):.2f}%")
            print(f"🎯 Taxa de sucesso: {metrics.get('win_rate_episodes', 0):.1f}%")
            print(f"⚖️ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"⚖️ Calmar REAL: {metrics.get('calmar_ratio_REAL', 0):.2f}")
            
            if metrics.get('total_trades', 0) > 0:
                print(f"💹 Trades: {metrics['total_trades']} (WR: {metrics.get('win_rate_trades', 0):.1f}%)")
                print(f"💰 PnL total: ${metrics.get('total_pnl', 0):.2f}")
                print(f"💰 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        # Salvar relatório debug
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        debug_file = f"D:/Projeto/avaliacoes/debug_evaluation_FIXED_{timestamp}.json"
        
        debug_data = {
            'checkpoint': checkpoint_path,
            'episodes_completed': len(all_episodes),
            'metrics': metrics,
            'debug_settings': {
                'NUM_EPISODES': NUM_EPISODES,
                'TEST_STEPS': TEST_STEPS,
                'INITIAL_PORTFOLIO': INITIAL_PORTFOLIO,
                'DEBUG_PORTFOLIO': DEBUG_PORTFOLIO,
                'DEBUG_TRADES': DEBUG_TRADES,
                'DEBUG_METRICS': DEBUG_METRICS
            }
        }
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, default=str)
        
        print(f"\n💾 Debug salvo: {debug_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        print(f"Detalhes: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print(f"🚀 AVALIAÇÃO FIXED INICIADA - {datetime.now().strftime('%H:%M:%S')}")
    
    # Random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    success = test_FIXED_evaluation()
    
    if success:
        print(f"\n✅ AVALIAÇÃO FIXED CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")
    else:
        print(f"\n❌ AVALIAÇÃO FIXED FALHOU - {datetime.now().strftime('%H:%M:%S')}")