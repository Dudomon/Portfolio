#!/usr/bin/env python3
"""
🎯 COMPARAÇÃO TRIPLA DAYTRADER: 7M vs 8M vs FINAL (10.8M)
Encontrar o melhor checkpoint do DayTrader
"""

import sys
import os
import traceback
from datetime import datetime
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch

# Configuração dos checkpoints DayTrader
DAYTRADER_7M = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase3noisehandling_7000000_steps_20250808_151427.zip"
DAYTRADER_8M = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase4stresstesting_8000000_steps_20250808_173027.zip"
DAYTRADER_FINAL = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/FINAL_ABSOLUTE_10836000_steps_20250808_234703.zip"

INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03
TEST_STEPS = 5000

def test_daytrader_checkpoint(checkpoint_path, checkpoint_name):
    """Testa um checkpoint específico do DayTrader"""
    
    print(f"\n🔥 TESTANDO {checkpoint_name}")
    print("=" * 50)
    
    try:
        # Imports
        from sb3_contrib import RecurrentPPO
        from daytrader import TradingEnv
        
        # Dataset real - usar SEMPRE a mesma seção
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
        
        # Usar seção fixa para comparação justa
        total_len = len(df)
        start_idx = total_len // 4  # Sempre mesma seção
        end_idx = start_idx + 8000   # 8k barras
        test_df = df.iloc[start_idx:end_idx]
        
        print(f"📊 Dataset: {len(test_df):,} barras ({test_df.index.min()} - {test_df.index.max()})")
        
        # Criar ambiente
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
        
        # Carregar modelo
        print(f"🤖 Carregando {checkpoint_name}...")
        model = RecurrentPPO.load(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        model.policy.set_training_mode(False)
        
        # EXECUTAR 5 RUNS para estabilidade
        all_results = []
        
        for run in range(5):
            print(f"  🔄 Run {run+1}/5...")
            
            obs = env.reset()
            lstm_states = None
            done = False
            step = 0
            
            # Tracking por run
            portfolio_history = [INITIAL_PORTFOLIO]
            trades_log = []
            entry_qualities = []
            actions_log = []
            
            while not done and step < TEST_STEPS:
                # Predição não-determinística
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                
                obs, reward, done, info = env.step(action)
                
                entry_quality = float(action[1])
                entry_qualities.append(entry_quality)
                
                actions_log.append({
                    'entry_decision': int(action[0]),
                    'entry_quality': entry_quality
                })
                
                # Log trades
                if hasattr(info, 'get') and info.get('trade_closed', False):
                    trades_log.append({
                        'pnl': info.get('trade_pnl', 0),
                        'type': info.get('trade_type', 'unknown'),
                        'entry_quality': entry_quality
                    })
                
                portfolio_history.append(env.portfolio_value)
                step += 1
            
            # Métricas do run
            final_portfolio = env.portfolio_value
            total_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
            
            avg_entry_quality = np.mean(entry_qualities)
            median_entry_quality = np.median(entry_qualities)
            std_entry_quality = np.std(entry_qualities)
            
            # Distribuição Entry Quality
            eq_array = np.array(entry_qualities)
            zeros_pct = (eq_array == 0.0).sum() / len(eq_array) * 100
            ones_pct = (eq_array == 1.0).sum() / len(eq_array) * 100
            middle_pct = 100 - zeros_pct - ones_pct
            
            # Trading stats
            total_trades = len(trades_log)
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            
            if total_trades > 0:
                profitable_trades = [t for t in trades_log if t['pnl'] > 0]
                losing_trades = [t for t in trades_log if t['pnl'] < 0]
                
                win_rate = (len(profitable_trades) / total_trades) * 100
                avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Ações
            decisions = [a['entry_decision'] for a in actions_log]
            hold_pct = (sum(1 for d in decisions if d == 0) / len(decisions)) * 100
            long_pct = (sum(1 for d in decisions if d == 1) / len(decisions)) * 100
            short_pct = (sum(1 for d in decisions if d == 2) / len(decisions)) * 100
            
            # Drawdown
            portfolio_array = np.array(portfolio_history)
            running_max = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - running_max) / running_max * 100
            max_drawdown = np.min(drawdown)
            
            run_result = {
                'run': run + 1,
                'final_portfolio': final_portfolio,
                'total_return': total_return,
                'avg_entry_quality': avg_entry_quality,
                'median_entry_quality': median_entry_quality,
                'std_entry_quality': std_entry_quality,
                'zeros_pct': zeros_pct,
                'ones_pct': ones_pct,
                'middle_pct': middle_pct,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'hold_pct': hold_pct,
                'long_pct': long_pct,
                'short_pct': short_pct,
                'max_drawdown': max_drawdown
            }
            
            all_results.append(run_result)
        
        # Calcular médias e desvios
        metrics = {}
        for key in all_results[0].keys():
            if key != 'run':
                values = [r[key] for r in all_results]
                metrics[f'{key}_avg'] = np.mean(values)
                metrics[f'{key}_std'] = np.std(values)
                metrics[f'{key}_min'] = np.min(values)
                metrics[f'{key}_max'] = np.max(values)
        
        metrics['checkpoint_name'] = checkpoint_name
        metrics['checkpoint_path'] = checkpoint_path
        metrics['individual_runs'] = all_results
        
        print(f"✅ {checkpoint_name} testado com sucesso!")
        return metrics
        
    except Exception as e:
        print(f"❌ ERRO testando {checkpoint_name}: {e}")
        print(f"Detalhes: {traceback.format_exc()}")
        return None

def compare_all_checkpoints(dt_7m, dt_8m, dt_final):
    """Comparação tripla completa"""
    
    print("\n" + "=" * 90)
    print("🏆 COMPARAÇÃO TRIPLA DAYTRADER: 7M vs 8M vs FINAL (10.8M)")
    print("=" * 90)
    
    print(f"\n📊 PERFORMANCE FINANCEIRA:")
    print(f"{'Métrica':<20} {'7M Steps':<12} {'8M Steps':<12} {'Final 10.8M':<12} {'🥇 Melhor':<10}")
    print("-" * 76)
    
    dt7_return = dt_7m['total_return_avg']
    dt8_return = dt_8m['total_return_avg']
    dtf_return = dt_final['total_return_avg']
    
    best_return = max(dt7_return, dt8_return, dtf_return)
    best_return_model = "7M" if dt7_return == best_return else "8M" if dt8_return == best_return else "Final"
    
    dt7_portfolio = dt_7m['final_portfolio_avg']
    dt8_portfolio = dt_8m['final_portfolio_avg']
    dtf_portfolio = dt_final['final_portfolio_avg']
    
    best_portfolio = max(dt7_portfolio, dt8_portfolio, dtf_portfolio)
    best_portfolio_model = "7M" if dt7_portfolio == best_portfolio else "8M" if dt8_portfolio == best_portfolio else "Final"
    
    dt7_dd = dt_7m['max_drawdown_avg']
    dt8_dd = dt_8m['max_drawdown_avg']
    dtf_dd = dt_final['max_drawdown_avg']
    
    best_dd = max(dt7_dd, dt8_dd, dtf_dd)  # Menos negativo é melhor
    best_dd_model = "7M" if dt7_dd == best_dd else "8M" if dt8_dd == best_dd else "Final"
    
    print(f"{'Retorno %':<20} {dt7_return:>+6.2f}%     {dt8_return:>+6.2f}%     {dtf_return:>+6.2f}%     {best_return_model}")
    print(f"{'Portfolio Final':<20} ${dt7_portfolio:>6.2f}    ${dt8_portfolio:>6.2f}    ${dtf_portfolio:>6.2f}    {best_portfolio_model}")
    print(f"{'Max Drawdown':<20} {dt7_dd:>6.2f}%     {dt8_dd:>6.2f}%     {dtf_dd:>6.2f}%     {best_dd_model}")
    
    print(f"\n🎯 ENTRY QUALITY ANALYSIS:")
    print(f"{'Métrica':<20} {'7M Steps':<12} {'8M Steps':<12} {'Final 10.8M':<12} {'🥇 Melhor':<10}")
    print("-" * 76)
    
    dt7_eq = dt_7m['avg_entry_quality_avg']
    dt8_eq = dt_8m['avg_entry_quality_avg']
    dtf_eq = dt_final['avg_entry_quality_avg']
    
    best_eq = max(dt7_eq, dt8_eq, dtf_eq)
    best_eq_model = "7M" if dt7_eq == best_eq else "8M" if dt8_eq == best_eq else "Final"
    
    dt7_med = dt_7m['median_entry_quality_avg']
    dt8_med = dt_8m['median_entry_quality_avg']
    dtf_med = dt_final['median_entry_quality_avg']
    
    best_med = max(dt7_med, dt8_med, dtf_med)
    best_med_model = "7M" if dt7_med == best_med else "8M" if dt8_med == best_med else "Final"
    
    dt7_std = dt_7m['std_entry_quality_avg']
    dt8_std = dt_8m['std_entry_quality_avg']
    dtf_std = dt_final['std_entry_quality_avg']
    
    best_std = min(dt7_std, dt8_std, dtf_std)  # Menor std é melhor
    best_std_model = "7M" if dt7_std == best_std else "8M" if dt8_std == best_std else "Final"
    
    print(f"{'EQ Média':<20} {dt7_eq:>7.3f}      {dt8_eq:>7.3f}      {dtf_eq:>7.3f}      {best_eq_model}")
    print(f"{'EQ Mediana':<20} {dt7_med:>7.3f}      {dt8_med:>7.3f}      {dtf_med:>7.3f}      {best_med_model}")
    print(f"{'EQ Std Dev':<20} {dt7_std:>7.3f}      {dt8_std:>7.3f}      {dtf_std:>7.3f}      {best_std_model}")
    
    print(f"\n📊 DISTRIBUIÇÃO ENTRY QUALITY:")
    print(f"{'Métrica':<20} {'7M Steps':<12} {'8M Steps':<12} {'Final 10.8M':<12} {'🥇 Melhor':<10}")
    print("-" * 76)
    
    dt7_zeros = dt_7m['zeros_pct_avg']
    dt8_zeros = dt_8m['zeros_pct_avg']
    dtf_zeros = dt_final['zeros_pct_avg']
    
    best_zeros = min(dt7_zeros, dt8_zeros, dtf_zeros)  # Menos zeros é melhor
    best_zeros_model = "7M" if dt7_zeros == best_zeros else "8M" if dt8_zeros == best_zeros else "Final"
    
    dt7_middle = dt_7m['middle_pct_avg']
    dt8_middle = dt_8m['middle_pct_avg']
    dtf_middle = dt_final['middle_pct_avg']
    
    best_middle = max(dt7_middle, dt8_middle, dtf_middle)  # Mais middle é melhor
    best_middle_model = "7M" if dt7_middle == best_middle else "8M" if dt8_middle == best_middle else "Final"
    
    print(f"{'% Zeros':<20} {dt7_zeros:>6.1f}%      {dt8_zeros:>6.1f}%      {dtf_zeros:>6.1f}%      {best_zeros_model}")
    print(f"{'% Middle':<20} {dt7_middle:>6.1f}%      {dt8_middle:>6.1f}%      {dtf_middle:>6.1f}%      {best_middle_model}")
    
    # SCORECARD FINAL
    print(f"\n🏆 SCORECARD FINAL:")
    print("=" * 40)
    
    scores = {"7M": 0, "8M": 0, "Final": 0}
    
    # Pontuação baseada em métricas importantes
    key_metrics = [
        (best_return_model, "Retorno"),
        (best_portfolio_model, "Portfolio"),
        (best_eq_model, "Entry Quality"),
        (best_med_model, "EQ Mediana"),
        (best_zeros_model, "Menos Zeros"),
        (best_middle_model, "Mais Middle"),
        (best_dd_model, "Drawdown")
    ]
    
    for winner, metric in key_metrics:
        scores[winner] += 1
        print(f"🥇 {winner}: {metric}")
    
    print(f"\n📊 PONTUAÇÃO TOTAL:")
    for model, score in scores.items():
        print(f"{model}: {score} pontos")
    
    # Determinar campeão
    champion = max(scores.items(), key=lambda x: x[1])
    runner_up = sorted(scores.items(), key=lambda x: x[1], reverse=True)[1]
    
    print(f"\n🏆 CAMPEÃO ABSOLUTO: DAYTRADER {champion[0].upper()}")
    print(f"🥈 Vice-campeão: DAYTRADER {runner_up[0].upper()}")
    
    # Análise vs Target Original
    print(f"\n🎯 ANÁLISE VS TARGET ORIGINAL (0.488):")
    target_original = 0.488
    
    dt7_vs_target = dt7_eq - target_original
    dt8_vs_target = dt8_eq - target_original
    dtf_vs_target = dtf_eq - target_original
    
    print(f"7M Steps: {dt7_eq:.3f} ({dt7_vs_target:+.3f})")
    print(f"8M Steps: {dt8_eq:.3f} ({dt8_vs_target:+.3f})")  
    print(f"Final: {dtf_eq:.3f} ({dtf_vs_target:+.3f})")
    
    # Determinar melhor vs target
    above_target = []
    if dt7_eq > target_original:
        above_target.append(("7M", dt7_eq, dt7_vs_target))
    if dt8_eq > target_original:
        above_target.append(("8M", dt8_eq, dt8_vs_target))
    if dtf_eq > target_original:
        above_target.append(("Final", dtf_eq, dtf_vs_target))
    
    if above_target:
        best_vs_target = max(above_target, key=lambda x: x[1])
        print(f"\n✅ MELHOR VS TARGET: DAYTRADER {best_vs_target[0]} ({best_vs_target[1]:.3f}, +{best_vs_target[2]:.3f})")
    else:
        print(f"\n❌ NENHUM MODELO SUPERA O TARGET ORIGINAL")
    
    # RECOMENDAÇÃO FINAL
    print(f"\n💡 RECOMENDAÇÃO FINAL:")
    print("=" * 30)
    
    champion_model = champion[0].upper()
    champion_eq = dt7_eq if champion[0] == "7M" else dt8_eq if champion[0] == "8M" else dtf_eq
    
    if champion_eq > target_original:
        print(f"✅ USE DAYTRADER {champion_model}")
        print(f"   Entry Quality: {champion_eq:.3f} (superior ao target)")
        print(f"   Campeão em {champion[1]} métricas")
    else:
        print(f"⚠️ DAYTRADER {champion_model} é o melhor disponível")
        print(f"   Mas Entry Quality: {champion_eq:.3f} (abaixo do target)")
        print(f"   Considere continuar treinamento")

def main():
    """Função principal"""
    
    print("🔥 COMPARAÇÃO TRIPLA DAYTRADER")
    print("=" * 50)
    print("🎯 Objetivo: Encontrar o MELHOR checkpoint")
    print("📊 Candidatos: 7M vs 8M vs Final (10.8M)")
    print("🔄 Teste: 5 runs x 5000 steps cada")
    print("🎖️ Target: Entry Quality > 0.488")
    print("=" * 50)
    
    # Testar todos os checkpoints
    dt_7m_metrics = test_daytrader_checkpoint(DAYTRADER_7M, "DAYTRADER 7M")
    dt_8m_metrics = test_daytrader_checkpoint(DAYTRADER_8M, "DAYTRADER 8M")
    dt_final_metrics = test_daytrader_checkpoint(DAYTRADER_FINAL, "DAYTRADER FINAL")
    
    if dt_7m_metrics and dt_8m_metrics and dt_final_metrics:
        # Comparação tripla
        compare_all_checkpoints(dt_7m_metrics, dt_8m_metrics, dt_final_metrics)
    else:
        print("❌ Falha ao testar um ou mais checkpoints")

if __name__ == "__main__":
    print(f"🚀 INICIANDO COMPARAÇÃO TRIPLA - {datetime.now().strftime('%H:%M:%S')}")
    main()
    print(f"\n✅ COMPARAÇÃO TRIPLA CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")