#!/usr/bin/env python3
"""
🔥 COMPARAÇÃO: LEGION vs DAYTRADER 3.45M
Usando o script avaliar_trading_real.py do usuário modificado para comparar os dois modelos
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch
from datetime import datetime
import traceback

# Configurações
LEGION_PATH = "D:/Projeto/Modelo PPO Trader/Modelo daytrade/Legion_extracted"
CHECKPOINT_345M_PATH = "D:/Projeto/trading_framework/training/checkpoints/DAYTRADER/checkpoint_3450000_steps_20250811_094827.zip"

INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03
TEST_STEPS = 3000

def test_trading_real(checkpoint_path, model_name):
    """🎯 Teste real de trading com configurações exatas do daytrader"""
    
    print(f"💰 TESTE TRADING REAL - {model_name}")
    print("=" * 60)
    print(f"💵 Portfolio Inicial: ${INITIAL_PORTFOLIO}")
    print(f"📊 Base Lot: {BASE_LOT_SIZE}")
    print(f"📊 Max Lot: {MAX_LOT_SIZE}")
    print(f"🧠 Modo: INFERÊNCIA (não-determinístico)")
    print("=" * 60)
    
    try:
        # Imports
        from sb3_contrib import RecurrentPPO
        from daytrader import TradingEnv
        
        print("✅ Imports carregados")
        
        # Dataset real para teste
        print("📊 Carregando dataset para teste...")
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
        
        # Pegar amostra do meio para evitar problemas
        total_len = len(df)
        start_idx = total_len // 2
        end_idx = start_idx + 5000  # 5k barras para teste
        test_df = df.iloc[start_idx:end_idx]
        
        print(f"✅ Dataset preparado: {len(test_df):,} barras")
        print(f"📅 Período: {test_df.index.min()} até {test_df.index.max()}")
        
        # Criar ambiente de trading com configurações EXATAS
        trading_params = {
            'base_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'initial_balance': INITIAL_PORTFOLIO,
            'target_trades_per_day': 18,  # Como no daytrader
            'stop_loss_range': (2.0, 8.0),
            'take_profit_range': (3.0, 15.0)
        }
        
        env = TradingEnv(
            test_df,
            window_size=20,
            is_training=False,  # 🔥 MODO AVALIAÇÃO
            initial_balance=INITIAL_PORTFOLIO,
            trading_params=trading_params
        )
        
        print("✅ Ambiente de trading configurado")
        
        # Carregar modelo
        print(f"🤖 Carregando modelo {model_name}...")
        
        if "LEGION" in model_name:
            # Para o Legion, carregar do diretório extraído
            # O Legion não é um arquivo ZIP, mas um diretório com arquivos .pth
            policy_path = os.path.join(checkpoint_path, "policy.pth")
            if os.path.exists(policy_path):
                from sb3_contrib import RecurrentPPO
                # Criar um modelo base e carregar os pesos manualmente
                print(f"⚠️ LEGION requer carregamento especial (arquivo .pth)")
                print(f"❌ Pulando LEGION - formato não compatível com RecurrentPPO.load()")
                return None
        else:
            model = RecurrentPPO.load(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            model.policy.set_training_mode(False)  # 🔥 MODO INFERÊNCIA
        
        print(f"✅ Modelo carregado em {model.device}")
        
        # EXECUTAR EPISÓDIO DE TRADING
        print(f"🚀 Iniciando episódio de trading ({TEST_STEPS} steps)...")
        
        obs = env.reset()
        lstm_states = None
        done = False
        step = 0
        
        # Variáveis de tracking
        portfolio_history = [INITIAL_PORTFOLIO]
        trades_log = []
        actions_log = []
        
        while not done and step < TEST_STEPS:
            # PREDIÇÃO EM MODO INFERÊNCIA (não-determinístico)
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
            
            # Executar ação no ambiente
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
        
        # ANÁLISE DE RESULTADOS
        print("\n" + "=" * 60)
        print("📊 RESULTADOS DO TRADING REAL")
        print("=" * 60)
        
        final_portfolio = env.portfolio_value
        total_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
        
        print(f"💵 Portfolio Inicial: ${INITIAL_PORTFOLIO:.2f}")
        print(f"💵 Portfolio Final: ${final_portfolio:.2f}")
        print(f"📈 Retorno Total: {total_return:+.2f}%")
        
        # Análise de trades
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
        total_pnl = 0
        profit_factor = 0
        
        if trades_log:
            total_trades = len(trades_log)
            profitable_trades = [t for t in trades_log if t['pnl'] > 0]
            losing_trades = [t for t in trades_log if t['pnl'] < 0]
            
            win_rate = (len(profitable_trades) / total_trades) * 100
            avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            total_pnl = sum(t['pnl'] for t in trades_log)
            
            print(f"📊 Total de Trades: {total_trades}")
            print(f"🎯 Win Rate: {win_rate:.1f}%")
            print(f"💚 Trades Lucrativos: {len(profitable_trades)}")
            print(f"❌ Trades Perdedores: {len(losing_trades)}")
            print(f"💰 Lucro Médio: ${avg_profit:.2f}")
            print(f"📉 Perda Média: ${avg_loss:.2f}")
            print(f"💵 PnL Total: ${total_pnl:.2f}")
            
            # Profit Factor
            if avg_loss != 0 and losing_trades:
                gross_profit = sum(t['pnl'] for t in profitable_trades)
                gross_loss = abs(sum(t['pnl'] for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                print(f"⚖️ Profit Factor: {profit_factor:.2f}")
            
            # Frequência de trading
            trading_frequency = (total_trades / step) * 100
            print(f"📈 Frequência de Trading: {trading_frequency:.2f}% dos steps")
            
        else:
            print("⚠️ NENHUM TRADE EXECUTADO")
            print("🔍 Modelo extremamente conservador")
        
        # Análise de ações
        hold_pct = long_pct = short_pct = avg_quality = 0
        if actions_log:
            entry_decisions = [a['entry_decision'] for a in actions_log]
            entry_qualities = [a['entry_quality'] for a in actions_log]
            
            hold_pct = (sum(1 for d in entry_decisions if d == 0) / len(entry_decisions)) * 100
            long_pct = (sum(1 for d in entry_decisions if d == 1) / len(entry_decisions)) * 100
            short_pct = (sum(1 for d in entry_decisions if d == 2) / len(entry_decisions)) * 100
            
            avg_quality = np.mean(entry_qualities)
            
            print(f"\n🎮 ANÁLISE DAS AÇÕES:")
            print(f"⚪ HOLD: {hold_pct:.1f}%")
            print(f"🟢 LONG: {long_pct:.1f}%")
            print(f"🔴 SHORT: {short_pct:.1f}%")
            print(f"⭐ Entry Quality Média: {avg_quality:.3f}")
        
        # Drawdown analysis
        max_drawdown = 0
        if len(portfolio_history) > 1:
            portfolio_array = np.array(portfolio_history)
            running_max = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - running_max) / running_max * 100
            max_drawdown = np.min(drawdown)
            
            print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        
        # Avaliação final
        print(f"\n🎖️ AVALIAÇÃO FINAL:")
        if total_return > 5:
            grade = "🟢 EXCELENTE"
        elif total_return > 2:
            grade = "🟡 BOM"
        elif total_return > -2:
            grade = "🟠 REGULAR"
        else:
            grade = "🔴 RUIM"
        
        print(f"   {grade}")
        print(f"   Retorno: {total_return:+.2f}%")
        if trades_log:
            print(f"   Trades: {len(trades_log)} (Win Rate: {win_rate:.1f}%)")
        
        # Retornar métricas para comparação
        return {
            'model_name': model_name,
            'final_portfolio': final_portfolio,
            'total_return_pct': total_return,
            'num_trades': len(trades_log),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'avg_entry_quality': avg_quality,
            'max_drawdown_pct': max_drawdown,
            'hold_pct': hold_pct,
            'long_pct': long_pct,
            'short_pct': short_pct
        }
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        print(f"Detalhes: {traceback.format_exc()}")
        return None

def compare_legion_vs_345m():
    """Comparar Legion vs Daytrader 3.45M usando o script do usuário"""
    print("🔥 COMPARAÇÃO LEGION vs DAYTRADER 3.45M")
    print("=" * 80)
    print(f"🕐 INICIANDO - {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Testar Legion
    print("1️⃣ TESTANDO LEGION...")
    legion_result = test_trading_real(LEGION_PATH, "LEGION")
    
    print("\n" + "🔄" * 20)
    
    # Testar Daytrader 3.45M
    print("2️⃣ TESTANDO DAYTRADER 3.45M...")
    daytrader_result = test_trading_real(CHECKPOINT_345M_PATH, "DAYTRADER 3.45M")
    
    if legion_result and daytrader_result:
        # COMPARAÇÃO FINAL
        print("\n" + "=" * 80)
        print("🏆 COMPARAÇÃO FINAL - LEGION vs DAYTRADER 3.45M")
        print("=" * 80)
        
        # Métricas chave para comparação
        metrics = [
            ('💵 Retorno Total', 'total_return_pct', '%', 'higher'),
            ('🔢 Número de Trades', 'num_trades', '', 'neutral'),
            ('🎯 Win Rate', 'win_rate', '%', 'higher'),
            ('⭐ Entry Quality', 'avg_entry_quality', '', 'higher'),
            ('📉 Max Drawdown', 'max_drawdown_pct', '%', 'lower'),
            ('💰 PnL Total', 'total_pnl', '$', 'higher'),
            ('⚖️ Profit Factor', 'profit_factor', '', 'higher')
        ]
        
        print(f"{'Métrica':<25} {'LEGION':<15} {'DAYTRADER':<15} {'Vencedor'}")
        print("-" * 75)
        
        legion_wins = 0
        daytrader_wins = 0
        
        for metric_name, metric_key, unit, comparison in metrics:
            legion_val = legion_result.get(metric_key, 0)
            daytrader_val = daytrader_result.get(metric_key, 0)
            
            # Determinar vencedor
            if comparison == 'higher':
                winner = "LEGION" if legion_val > daytrader_val else "DAYTRADER" if daytrader_val > legion_val else "EMPATE"
                if legion_val > daytrader_val:
                    legion_wins += 1
                elif daytrader_val > legion_val:
                    daytrader_wins += 1
            elif comparison == 'lower':
                winner = "LEGION" if legion_val < daytrader_val else "DAYTRADER" if daytrader_val < legion_val else "EMPATE"
                if legion_val < daytrader_val:
                    legion_wins += 1
                elif daytrader_val < legion_val:
                    daytrader_wins += 1
            else:  # neutral
                winner = "INFO"
            
            if unit == '$':
                print(f"{metric_name:<25} ${legion_val:<14.2f} ${daytrader_val:<14.2f} {'🏆 ' + winner if winner not in ['INFO', 'EMPATE'] else winner}")
            else:
                print(f"{metric_name:<25} {legion_val:<14.2f}{unit} {daytrader_val:<14.2f}{unit} {'🏆 ' + winner if winner not in ['INFO', 'EMPATE'] else winner}")
        
        print("\n" + "=" * 80)
        print("🏆 RESULTADO FINAL:")
        
        if legion_wins > daytrader_wins:
            print(f"   🥇 LEGION VENCEU: {legion_wins} x {daytrader_wins}")
            print(f"   🎉 LEGION é SUPERIOR ao Daytrader 3.45M!")
        elif daytrader_wins > legion_wins:
            print(f"   🥇 DAYTRADER 3.45M VENCEU: {daytrader_wins} x {legion_wins}")
            print(f"   🎉 DAYTRADER 3.45M é SUPERIOR ao Legion!")
        else:
            print(f"   🤝 EMPATE: {legion_wins} x {daytrader_wins}")
            print(f"   ⚖️ Modelos têm performance equivalente")
        
        # Resumo executivo
        print(f"\n📋 RESUMO EXECUTIVO:")
        print(f"   Legion: {legion_result['total_return_pct']:+.2f}% retorno, {legion_result['num_trades']} trades")
        print(f"   Daytrader 3.45M: {daytrader_result['total_return_pct']:+.2f}% retorno, {daytrader_result['num_trades']} trades")
        
        return {
            'legion': legion_result,
            'daytrader': daytrader_result,
            'winner': 'LEGION' if legion_wins > daytrader_wins else 'DAYTRADER' if daytrader_wins > legion_wins else 'EMPATE'
        }
    
    else:
        print("❌ Não foi possível completar a comparação")
        return None

if __name__ == "__main__":
    results = compare_legion_vs_345m()
    print(f"\n🕐 COMPARAÇÃO CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")