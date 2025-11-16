#!/usr/bin/env python3
"""
🔥 COMPARAÇÃO: LEGION vs DAYTRADER 3.45M
Usar o script de avaliação para comparar os dois modelos
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
TEST_STEPS = 2000  # Reduzido para ser mais rápido

def test_model(model_path, model_name, is_legion=False):
    """Testar um modelo específico"""
    print(f"\n🤖 TESTANDO: {model_name}")
    print("=" * 60)
    
    try:
        # Imports
        from sb3_contrib import RecurrentPPO
        from daytrader import TradingEnv
        
        # Dataset real para teste
        print("📊 Carregando dataset para teste...")
        dataset_path = "D:/Projeto/data/GC_YAHOO_ENHANCED_V3_BALANCED_20250804_192226.csv"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset não encontrado: {dataset_path}")
            return None
            
        df = pd.read_csv(dataset_path, parse_dates=['datetime'], index_col='datetime')
        df = df.tail(3000)  # Usar últimas 3000 barras
        print(f"✅ Dataset preparado: {len(df):,} barras")
        print(f"📅 Período: {df.index[0]} até {df.index[-1]}")
        
        # Criar ambiente
        env = TradingEnv(
            df=df,
            window_size=20,
            is_training=False,  # Modo avaliação
            initial_balance=INITIAL_PORTFOLIO,
            trading_params={
                'base_lot': BASE_LOT_SIZE,
                'max_lot': MAX_LOT_SIZE
            }
        )
        print("✅ Ambiente de trading configurado")
        
        # Carregar modelo
        print(f"🤖 Carregando modelo {model_name}...")
        
        if is_legion:
            # Carregar Legion diretamente do diretório extraído
            model = RecurrentPPO.load(model_path, env=env, device='cuda')
        else:
            # Carregar checkpoint do daytrader
            model = RecurrentPPO.load(model_path, env=env, device='cuda')
        
        print("✅ Modelo carregado em cuda")
        
        # Executar teste
        print(f"🚀 Iniciando episódio de trading ({TEST_STEPS} steps)...")
        
        obs = env.reset()
        total_rewards = []
        actions_count = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
        entry_qualities = []
        
        for step in range(TEST_STEPS):
            # Predição
            action, _states = model.predict(obs, deterministic=False)
            
            # Registrar ação
            action_type = int(action[0])
            if action_type == 0:
                actions_count['HOLD'] += 1
            elif action_type == 1:
                actions_count['LONG'] += 1
            else:
                actions_count['SHORT'] += 1
                
            # Registrar entry quality
            entry_qualities.append(float(action[1]))
            
            # Executar step
            obs, reward, done, info = env.step(action)
            total_rewards.append(reward)
            
            # Progress
            if (step + 1) % 500 == 0:
                print(f"  Step {step + 1}/{TEST_STEPS} - Portfolio: ${env.portfolio_value:.2f}")
            
            if done:
                obs = env.reset()
        
        # Calcular métricas
        final_portfolio = env.portfolio_value
        total_return = (final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO * 100
        num_trades = len(env.trades)
        avg_entry_quality = np.mean(entry_qualities)
        max_drawdown = env.peak_drawdown * 100
        
        # Distribuição de ações
        total_actions = sum(actions_count.values())
        action_percentages = {k: v/total_actions*100 for k, v in actions_count.items()}
        
        result = {
            'model_name': model_name,
            'initial_portfolio': INITIAL_PORTFOLIO,
            'final_portfolio': final_portfolio,
            'total_return_pct': total_return,
            'num_trades': num_trades,
            'avg_entry_quality': avg_entry_quality,
            'max_drawdown_pct': max_drawdown,
            'actions': action_percentages,
            'avg_reward': np.mean(total_rewards),
            'total_reward': np.sum(total_rewards)
        }
        
        # Exibir resultados
        print(f"\n📊 RESULTADOS - {model_name}")
        print("=" * 50)
        print(f"💵 Portfolio Inicial: ${INITIAL_PORTFOLIO:.2f}")
        print(f"💵 Portfolio Final: ${final_portfolio:.2f}")
        print(f"📈 Retorno Total: {total_return:+.2f}%")
        print(f"🔢 Total de Trades: {num_trades}")
        print(f"⭐ Entry Quality Média: {avg_entry_quality:.3f}")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"🎮 Distribuição de Ações:")
        for action, pct in action_percentages.items():
            print(f"   {action}: {pct:.1f}%")
        print(f"🏆 Reward Médio: {np.mean(total_rewards):.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ ERRO ao testar {model_name}: {e}")
        traceback.print_exc()
        return None

def compare_models():
    """Comparar Legion vs Daytrader 3.45M"""
    print("🔥 COMPARAÇÃO COMPLETA: LEGION vs DAYTRADER 3.45M")
    print("=" * 80)
    
    # Testar Legion
    legion_result = test_model(LEGION_PATH, "LEGION", is_legion=True)
    
    # Testar Daytrader 3.45M  
    daytrader_result = test_model(CHECKPOINT_345M_PATH, "DAYTRADER 3.45M", is_legion=False)
    
    if legion_result and daytrader_result:
        # Comparação final
        print(f"\n🏆 COMPARAÇÃO FINAL")
        print("=" * 80)
        
        comparison_metrics = [
            ('💵 Retorno Total', 'total_return_pct', '%'),
            ('🔢 Número de Trades', 'num_trades', ''),
            ('⭐ Entry Quality', 'avg_entry_quality', ''),
            ('📉 Max Drawdown', 'max_drawdown_pct', '%'),
            ('🏆 Reward Médio', 'avg_reward', ''),
        ]
        
        print(f"{'Métrica':<20} {'LEGION':<15} {'DAYTRADER 3.45M':<18} {'Vencedor'}")
        print("-" * 70)
        
        legion_wins = 0
        daytrader_wins = 0
        
        for metric_name, metric_key, unit in comparison_metrics:
            legion_val = legion_result[metric_key]
            daytrader_val = daytrader_result[metric_key]
            
            # Determinar vencedor (maior é melhor, exceto drawdown)
            if metric_key == 'max_drawdown_pct':
                winner = "LEGION" if legion_val < daytrader_val else "DAYTRADER"
                if legion_val < daytrader_val:
                    legion_wins += 1
                else:
                    daytrader_wins += 1
            else:
                winner = "LEGION" if legion_val > daytrader_val else "DAYTRADER"
                if legion_val > daytrader_val:
                    legion_wins += 1
                else:
                    daytrader_wins += 1
            
            print(f"{metric_name:<20} {legion_val:<15.2f}{unit} {daytrader_val:<18.2f}{unit} {'🏆 ' + winner if winner else ''}")
        
        print("\n" + "=" * 80)
        print(f"🏆 RESULTADO FINAL:")
        if legion_wins > daytrader_wins:
            print(f"   LEGION VENCEU: {legion_wins} x {daytrader_wins}")
            print(f"   🎉 LEGION é SUPERIOR ao Daytrader 3.45M!")
        elif daytrader_wins > legion_wins:
            print(f"   DAYTRADER 3.45M VENCEU: {daytrader_wins} x {legion_wins}")  
            print(f"   🎉 DAYTRADER 3.45M é SUPERIOR ao Legion!")
        else:
            print(f"   EMPATE: {legion_wins} x {daytrader_wins}")
            print(f"   🤝 Modelos têm performance equivalente")
        
        # Diferenças percentuais
        print(f"\n📊 DIFERENÇAS CHAVE:")
        return_diff = ((daytrader_result['total_return_pct'] - legion_result['total_return_pct']) / abs(legion_result['total_return_pct']) * 100) if legion_result['total_return_pct'] != 0 else 0
        quality_diff = ((daytrader_result['avg_entry_quality'] - legion_result['avg_entry_quality']) / legion_result['avg_entry_quality'] * 100) if legion_result['avg_entry_quality'] != 0 else 0
        
        print(f"   Retorno: Daytrader {return_diff:+.1f}% vs Legion")
        print(f"   Entry Quality: Daytrader {quality_diff:+.1f}% vs Legion")
        
        return {
            'legion': legion_result,
            'daytrader': daytrader_result,
            'winner': 'LEGION' if legion_wins > daytrader_wins else 'DAYTRADER' if daytrader_wins > legion_wins else 'EMPATE'
        }
    
    else:
        print("❌ Não foi possível completar a comparação")
        return None

if __name__ == "__main__":
    print(f"🕐 INICIANDO COMPARAÇÃO - {datetime.now().strftime('%H:%M:%S')}")
    results = compare_models()
    print(f"🕐 COMPARAÇÃO CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")