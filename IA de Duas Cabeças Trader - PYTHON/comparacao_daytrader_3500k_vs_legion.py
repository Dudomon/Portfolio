#!/usr/bin/env python3
"""
🎯 COMPARAÇÃO TRADING REAL - DAYTRADER 3.5M vs LEGION
Avaliação comparativa dos dois modelos em condições reais de trading
"""

import sys
import os
import traceback
import zipfile
import tempfile
from datetime import datetime
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch

# Configurações dos modelos
DAYTRADER_3500K_PATH = "D:/Projeto/trading_framework/training/checkpoints/DAYTRADER/checkpoint_3500000_steps_20250811_095508.zip"
LEGION_PATH = "D:/Projeto/Modelo PPO Trader/Modelo daytrade/Legion daytrade.zip"

# Configurações de teste
INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03
TEST_STEPS = 3000

def extract_and_load_model(zip_path, model_name):
    """Extrai e carrega modelo do ZIP"""
    print(f"📦 Extraindo modelo {model_name} de {zip_path}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Procurar arquivo policy
            policy_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file == 'policy' or file == 'policy.zip':
                        policy_files.append(os.path.join(root, file))
            
            if not policy_files:
                print(f"❌ Arquivo policy não encontrado em {zip_path}")
                return None
            
            policy_path = policy_files[0]
            print(f"✅ Policy encontrado: {policy_path}")
            
            # Carregar modelo
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(policy_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            model.policy.set_training_mode(False)
            
            print(f"✅ Modelo {model_name} carregado em {model.device}")
            return model
            
    except Exception as e:
        print(f"❌ Erro ao carregar {model_name}: {e}")
        return None

def prepare_dataset():
    """Prepara dataset para teste"""
    print("📊 Preparando dataset...")
    
    dataset_path = "D:/Projeto/data/GC_YAHOO_ENHANCED_V3_BALANCED_20250804_192226.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return None
        
    df = pd.read_csv(dataset_path)
    
    # Processar dataset
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
        df.set_index('timestamp', inplace=True)
        df.drop('time', axis=1, inplace=True)
    
    # Renomear colunas
    df = df.rename(columns={
        'open': 'open_5m',
        'high': 'high_5m',
        'low': 'low_5m', 
        'close': 'close_5m',
        'tick_volume': 'volume_5m'
    })
    
    # Pegar amostra do meio
    total_len = len(df)
    start_idx = total_len // 2
    end_idx = start_idx + 5000
    test_df = df.iloc[start_idx:end_idx]
    
    print(f"✅ Dataset preparado: {len(test_df):,} barras")
    print(f"📅 Período: {test_df.index.min()} até {test_df.index.max()}")
    
    return test_df

def test_model(model, model_name, test_df):
    """Testa um modelo específico"""
    print(f"\n{'='*60}")
    print(f"🤖 TESTANDO MODELO: {model_name}")
    print(f"{'='*60}")
    
    try:
        from daytrader import TradingEnv
        
        # Configurar ambiente
        trading_params = {
            'base_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'initial_balance': INITIAL_PORTFOLIO,
            'target_trades_per_day': 18,
            'stop_loss_range': (2.0, 8.0),
            'take_profit_range': (3.0, 15.0)
        }
        
        env = TradingEnv(
            test_df,
            window_size=20,
            is_training=False,
            initial_balance=INITIAL_PORTFOLIO,
            trading_params=trading_params
        )
        
        # Executar episódio
        obs = env.reset()
        lstm_states = None
        done = False
        step = 0
        
        portfolio_history = [INITIAL_PORTFOLIO]
        trades_log = []
        actions_log = []
        
        print(f"🚀 Iniciando trading ({TEST_STEPS} steps)...")
        
        while not done and step < TEST_STEPS:
            # Predição
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
            
            # Executar ação
            obs, reward, done, info = env.step(action)
            
            # Log da ação
            actions_log.append({
                'step': step,
                'entry_decision': int(action[0]),
                'entry_quality': float(action[1]),
                'portfolio_value': env.portfolio_value,
                'current_price': getattr(env, 'current_price', 0)
            })
            
            # Log de trades
            if hasattr(info, 'get') and info.get('trade_closed', False):
                trade_info = {
                    'step': step,
                    'type': info.get('trade_type', 'unknown'),
                    'entry_price': info.get('entry_price', 0),
                    'exit_price': info.get('exit_price', 0),
                    'pnl': info.get('trade_pnl', 0),
                    'lot_size': info.get('lot_size', 0),
                    'duration': info.get('trade_duration', 0)
                }
                trades_log.append(trade_info)
                print(f"  💼 Trade #{len(trades_log)}: {trade_info['type']} PnL=${trade_info['pnl']:.2f}")
            
            portfolio_history.append(env.portfolio_value)
            
            if (step + 1) % 500 == 0:
                print(f"  Step {step+1}/{TEST_STEPS} - Portfolio: ${env.portfolio_value:.2f}")
            
            step += 1
        
        # Calcular métricas
        final_portfolio = env.portfolio_value
        total_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
        
        # Análise de trades
        total_trades = len(trades_log)
        profitable_trades = [t for t in trades_log if t['pnl'] > 0]
        losing_trades = [t for t in trades_log if t['pnl'] < 0]
        
        win_rate = (len(profitable_trades) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(t['pnl'] for t in trades_log)
        
        # Profit Factor
        if losing_trades:
            gross_profit = sum(t['pnl'] for t in profitable_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = float('inf') if profitable_trades else 0
        
        # Drawdown
        if len(portfolio_history) > 1:
            portfolio_array = np.array(portfolio_history)
            running_max = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - running_max) / running_max * 100
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0
        
        # Frequência de trading
        trading_frequency = (total_trades / step) * 100 if step > 0 else 0
        
        # Análise de ações
        if actions_log:
            entry_decisions = [a['entry_decision'] for a in actions_log]
            entry_qualities = [a['entry_quality'] for a in actions_log]
            
            hold_pct = (sum(1 for d in entry_decisions if d == 0) / len(entry_decisions)) * 100
            long_pct = (sum(1 for d in entry_decisions if d == 1) / len(entry_decisions)) * 100
            short_pct = (sum(1 for d in entry_decisions if d == 2) / len(entry_decisions)) * 100
            avg_quality = np.mean(entry_qualities)
        else:
            hold_pct = long_pct = short_pct = avg_quality = 0
        
        results = {
            'model_name': model_name,
            'final_portfolio': final_portfolio,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trading_frequency': trading_frequency,
            'hold_pct': hold_pct,
            'long_pct': long_pct,
            'short_pct': short_pct,
            'avg_entry_quality': avg_quality,
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades)
        }
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS {model_name}:")
        print(f"💵 Portfolio Final: ${final_portfolio:.2f}")
        print(f"📈 Retorno Total: {total_return:+.2f}%")
        print(f"📊 Total Trades: {total_trades}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"⚖️ Profit Factor: {profit_factor:.2f}")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"📈 Trading Freq: {trading_frequency:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ Erro no teste {model_name}: {e}")
        print(f"Detalhes: {traceback.format_exc()}")
        return None

def compare_models():
    """Compara os dois modelos"""
    print("🎯 COMPARAÇÃO DAYTRADER 3.5M vs LEGION")
    print("=" * 80)
    
    # Preparar dataset
    test_df = prepare_dataset()
    if test_df is None:
        return False
    
    # Carregar modelos
    print("\n📦 CARREGANDO MODELOS...")
    daytrader_model = extract_and_load_model(DAYTRADER_3500K_PATH, "DAYTRADER 3.5M")
    legion_model = extract_and_load_model(LEGION_PATH, "LEGION")
    
    if daytrader_model is None or legion_model is None:
        print("❌ Falha ao carregar modelos")
        return False
    
    # Testar modelos
    daytrader_results = test_model(daytrader_model, "DAYTRADER 3.5M", test_df)
    legion_results = test_model(legion_model, "LEGION", test_df)
    
    if daytrader_results is None or legion_results is None:
        print("❌ Falha nos testes")
        return False
    
    # COMPARAÇÃO FINAL
    print(f"\n{'='*80}")
    print("🏆 COMPARAÇÃO FINAL")
    print(f"{'='*80}")
    
    print(f"{'Métrica':<20} {'DAYTRADER 3.5M':<20} {'LEGION':<20} {'Vencedor':<15}")
    print("-" * 80)
    
    metrics = [
        ('Retorno Total', 'total_return', '%', 'higher'),
        ('Total Trades', 'total_trades', '', 'higher'),
        ('Win Rate', 'win_rate', '%', 'higher'),
        ('Profit Factor', 'profit_factor', '', 'higher'),
        ('Max Drawdown', 'max_drawdown', '%', 'lower'),
        ('Trading Freq', 'trading_frequency', '%', 'optimal'),
        ('Entry Quality', 'avg_entry_quality', '', 'higher')
    ]
    
    daytrader_wins = 0
    legion_wins = 0
    
    for metric_name, metric_key, unit, comparison in metrics:
        dt_val = daytrader_results[metric_key]
        lg_val = legion_results[metric_key]
        
        if comparison == 'higher':
            winner = "DAYTRADER" if dt_val > lg_val else "LEGION" if lg_val > dt_val else "EMPATE"
        elif comparison == 'lower':
            winner = "DAYTRADER" if dt_val < lg_val else "LEGION" if lg_val < dt_val else "EMPATE"
        else:  # optimal (trading freq around 5-15%)
            dt_opt = abs(dt_val - 10)  # Distância do ideal (10%)
            lg_opt = abs(lg_val - 10)
            winner = "DAYTRADER" if dt_opt < lg_opt else "LEGION" if lg_opt < dt_opt else "EMPATE"
        
        if winner == "DAYTRADER":
            daytrader_wins += 1
        elif winner == "LEGION":
            legion_wins += 1
        
        print(f"{metric_name:<20} {dt_val:<19.2f}{unit} {lg_val:<19.2f}{unit} {winner:<15}")
    
    print("-" * 80)
    print(f"🏆 PONTUAÇÃO FINAL:")
    print(f"   DAYTRADER 3.5M: {daytrader_wins} vitórias")
    print(f"   LEGION: {legion_wins} vitórias")
    
    overall_winner = "DAYTRADER 3.5M" if daytrader_wins > legion_wins else "LEGION" if legion_wins > daytrader_wins else "EMPATE"
    print(f"   🥇 VENCEDOR GERAL: {overall_winner}")
    
    # Análise detalhada
    print(f"\n📋 ANÁLISE DETALHADA:")
    print(f"DAYTRADER 3.5M:")
    print(f"  • Retorno: {daytrader_results['total_return']:+.2f}%")
    print(f"  • Trades: {daytrader_results['total_trades']} (Win: {daytrader_results['win_rate']:.1f}%)")
    print(f"  • Risco: Max DD {daytrader_results['max_drawdown']:.2f}%")
    
    print(f"\nLEGION:")
    print(f"  • Retorno: {legion_results['total_return']:+.2f}%")
    print(f"  • Trades: {legion_results['total_trades']} (Win: {legion_results['win_rate']:.1f}%)")
    print(f"  • Risco: Max DD {legion_results['max_drawdown']:.2f}%")
    
    # Recomendação
    print(f"\n💡 RECOMENDAÇÃO:")
    if overall_winner == "DAYTRADER 3.5M":
        print(f"   O DAYTRADER 3.5M apresentou melhor desempenho geral")
    elif overall_winner == "LEGION":
        print(f"   O LEGION apresentou melhor desempenho geral")
    else:
        print(f"   Ambos os modelos apresentaram desempenho similar")
    
    return True

if __name__ == "__main__":
    print(f"🚀 INICIANDO COMPARAÇÃO - {datetime.now().strftime('%H:%M:%S')}")
    
    success = compare_models()
    
    if success:
        print(f"\n✅ COMPARAÇÃO CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")
    else:
        print(f"\n❌ COMPARAÇÃO FALHOU - {datetime.now().strftime('%H:%M:%S')}")