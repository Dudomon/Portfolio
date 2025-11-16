#!/usr/bin/env python3
"""
🎯 COMPARAÇÃO FINAL: EXPERTGAIN V2 vs DAYTRADER
Análise completa lado a lado dos checkpoints finais
"""

import sys
import os
import traceback
from datetime import datetime
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch

# Configuração dos checkpoints
EXPERTGAIN_CHECKPOINT = "D:/Projeto/Otimizacao/treino_principal/models/EXPERTGAIN_V2/FINAL_expertgainv2phase2calibrate_1250000_steps_20250809_201727.zip"
DAYTRADER_CHECKPOINT = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/FINAL_ABSOLUTE_10836000_steps_20250808_234703.zip"

INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03
TEST_STEPS = 5000  # Teste mais longo para mais precisão

def test_model(checkpoint_path, model_name):
    """Testa um modelo específico"""
    
    print(f"\n🔥 TESTANDO {model_name}")
    print("=" * 50)
    
    try:
        # Imports
        from sb3_contrib import RecurrentPPO
        from daytrader import TradingEnv
        
        # Dataset real
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
        
        # Usar sempre a mesma seção para comparação justa
        total_len = len(df)
        start_idx = total_len // 3  # Terço inicial para consistência
        end_idx = start_idx + 8000   # 8k barras para teste mais robusto
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
        print(f"🤖 Carregando {model_name}...")
        model = RecurrentPPO.load(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        model.policy.set_training_mode(False)
        
        # EXECUTAR 3 RUNS para média mais estável
        all_results = []
        
        for run in range(3):
            print(f"  🔄 Run {run+1}/3...")
            
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
                # Predição não-determinística para variabilidade
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
            
            # Calcular métricas do run
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
            profit_factor = 0
            
            if total_trades > 0:
                profitable_trades = [t for t in trades_log if t['pnl'] > 0]
                losing_trades = [t for t in trades_log if t['pnl'] < 0]
                
                win_rate = (len(profitable_trades) / total_trades) * 100
                avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
                
                gross_profit = sum(t['pnl'] for t in profitable_trades)
                gross_loss = abs(sum(t['pnl'] for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Decisões
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
                'profit_factor': profit_factor,
                'hold_pct': hold_pct,
                'long_pct': long_pct,
                'short_pct': short_pct,
                'max_drawdown': max_drawdown
            }
            
            all_results.append(run_result)
        
        # Calcular médias das 3 runs
        metrics = {}
        for key in all_results[0].keys():
            if key != 'run':
                values = [r[key] for r in all_results]
                metrics[f'{key}_avg'] = np.mean(values)
                metrics[f'{key}_std'] = np.std(values)
        
        metrics['model_name'] = model_name
        metrics['checkpoint_path'] = checkpoint_path
        metrics['individual_runs'] = all_results
        
        print(f"✅ {model_name} testado com sucesso!")
        return metrics
        
    except Exception as e:
        print(f"❌ ERRO testando {model_name}: {e}")
        print(f"Detalhes: {traceback.format_exc()}")
        return None

def compare_models(expertgain_metrics, daytrader_metrics):
    """Comparação detalhada lado a lado"""
    
    print("\n" + "=" * 80)
    print("🏆 COMPARAÇÃO FINAL: EXPERTGAIN V2 vs DAYTRADER")
    print("=" * 80)
    
    print(f"\n📊 PERFORMANCE FINANCEIRA:")
    print(f"{'Métrica':<20} {'ExpertGain V2':<15} {'DayTrader':<15} {'Diferença':<15}")
    print("-" * 65)
    
    eg_return = expertgain_metrics['total_return_avg']
    dt_return = daytrader_metrics['total_return_avg']
    return_diff = eg_return - dt_return
    
    eg_portfolio = expertgain_metrics['final_portfolio_avg']
    dt_portfolio = daytrader_metrics['final_portfolio_avg']
    portfolio_diff = eg_portfolio - dt_portfolio
    
    eg_dd = expertgain_metrics['max_drawdown_avg']
    dt_dd = daytrader_metrics['max_drawdown_avg']
    dd_diff = eg_dd - dt_dd
    
    print(f"{'Retorno %':<20} {eg_return:>+7.2f}%      {dt_return:>+7.2f}%      {return_diff:>+7.2f}%")
    print(f"{'Portfolio Final':<20} ${eg_portfolio:>7.2f}      ${dt_portfolio:>7.2f}      ${portfolio_diff:>+7.2f}")
    print(f"{'Max Drawdown':<20} {eg_dd:>7.2f}%      {dt_dd:>7.2f}%      {dd_diff:>+7.2f}%")
    
    print(f"\n🎯 ENTRY QUALITY ANALYSIS:")
    print(f"{'Métrica':<20} {'ExpertGain V2':<15} {'DayTrader':<15} {'Diferença':<15}")
    print("-" * 65)
    
    eg_eq = expertgain_metrics['avg_entry_quality_avg']
    dt_eq = daytrader_metrics['avg_entry_quality_avg']
    eq_diff = eg_eq - dt_eq
    
    eg_med = expertgain_metrics['median_entry_quality_avg']
    dt_med = daytrader_metrics['median_entry_quality_avg']
    med_diff = eg_med - dt_med
    
    eg_std = expertgain_metrics['std_entry_quality_avg']
    dt_std = daytrader_metrics['std_entry_quality_avg']
    std_diff = eg_std - dt_std
    
    print(f"{'EQ Média':<20} {eg_eq:>7.3f}        {dt_eq:>7.3f}        {eq_diff:>+7.3f}")
    print(f"{'EQ Mediana':<20} {eg_med:>7.3f}        {dt_med:>7.3f}        {med_diff:>+7.3f}")
    print(f"{'EQ Std Dev':<20} {eg_std:>7.3f}        {dt_std:>7.3f}        {std_diff:>+7.3f}")
    
    print(f"\n📊 DISTRIBUIÇÃO ENTRY QUALITY:")
    print(f"{'Métrica':<20} {'ExpertGain V2':<15} {'DayTrader':<15} {'Diferença':<15}")
    print("-" * 65)
    
    eg_zeros = expertgain_metrics['zeros_pct_avg']
    dt_zeros = daytrader_metrics['zeros_pct_avg']
    zeros_diff = eg_zeros - dt_zeros
    
    eg_ones = expertgain_metrics['ones_pct_avg']
    dt_ones = daytrader_metrics['ones_pct_avg']
    ones_diff = eg_ones - dt_ones
    
    eg_middle = expertgain_metrics['middle_pct_avg']
    dt_middle = daytrader_metrics['middle_pct_avg']
    middle_diff = eg_middle - dt_middle
    
    print(f"{'% Zeros':<20} {eg_zeros:>7.1f}%        {dt_zeros:>7.1f}%        {zeros_diff:>+7.1f}%")
    print(f"{'% Ones':<20} {eg_ones:>7.1f}%        {dt_ones:>7.1f}%        {ones_diff:>+7.1f}%")
    print(f"{'% Middle':<20} {eg_middle:>7.1f}%        {dt_middle:>7.1f}%        {middle_diff:>+7.1f}%")
    
    print(f"\n💼 TRADING BEHAVIOR:")
    print(f"{'Métrica':<20} {'ExpertGain V2':<15} {'DayTrader':<15} {'Diferença':<15}")
    print("-" * 65)
    
    eg_trades = expertgain_metrics['total_trades_avg']
    dt_trades = daytrader_metrics['total_trades_avg']
    trades_diff = eg_trades - dt_trades
    
    eg_wr = expertgain_metrics['win_rate_avg']
    dt_wr = daytrader_metrics['win_rate_avg']
    wr_diff = eg_wr - dt_wr
    
    eg_hold = expertgain_metrics['hold_pct_avg']
    dt_hold = daytrader_metrics['hold_pct_avg']
    hold_diff = eg_hold - dt_hold
    
    print(f"{'Total Trades':<20} {eg_trades:>7.1f}        {dt_trades:>7.1f}        {trades_diff:>+7.1f}")
    print(f"{'Win Rate %':<20} {eg_wr:>7.1f}%        {dt_wr:>7.1f}%        {wr_diff:>+7.1f}%")
    print(f"{'% Hold':<20} {eg_hold:>7.1f}%        {dt_hold:>7.1f}%        {hold_diff:>+7.1f}%")
    
    # VEREDICTO FINAL
    print(f"\n🏆 VEREDICTO FINAL:")
    print("=" * 50)
    
    # Target original: 0.488 → 0.55+
    target_original = 0.488
    target_expertgain = 0.55
    
    print(f"🎯 OBJETIVO EXPERTGAIN:")
    print(f"   DayTrader Original: {target_original:.3f}")
    print(f"   Target ExpertGain: {target_expertgain:.3f}+")
    print(f"   Melhoria Necessária: +{target_expertgain - target_original:.3f}")
    
    print(f"\n📈 RESULTADOS OBTIDOS:")
    print(f"   DayTrader Final: {dt_eq:.3f}")
    print(f"   ExpertGain V2: {eg_eq:.3f}")
    print(f"   Diferença Real: {eq_diff:+.3f}")
    
    # Análise de sucesso
    expertgain_success = eg_eq >= target_expertgain
    improvement_over_original = eg_eq > target_original
    improvement_over_daytrader = eg_eq > dt_eq
    
    print(f"\n🔍 ANÁLISE DE SUCESSO:")
    
    if expertgain_success:
        print(f"   ✅ OBJETIVO ATINGIDO: {eg_eq:.3f} ≥ {target_expertgain}")
        final_verdict = "🟢 EXPERTGAIN V2 SUCESSO COMPLETO"
    elif improvement_over_daytrader and eg_eq >= target_original:
        print(f"   🟡 MELHORIA PARCIAL: {eg_eq:.3f} > {dt_eq:.3f} (DayTrader)")
        print(f"   🟡 ACIMA DO ORIGINAL: {eg_eq:.3f} > {target_original} (Original)")
        final_verdict = "🟡 EXPERTGAIN V2 MELHORIA PARCIAL"
    elif improvement_over_daytrader:
        print(f"   🟡 MELHOR QUE DAYTRADER: {eg_eq:.3f} > {dt_eq:.3f}")
        print(f"   ❌ ABAIXO DO ORIGINAL: {eg_eq:.3f} < {target_original}")
        final_verdict = "🟡 EXPERTGAIN V2 MELHORIA LIMITADA"
    else:
        print(f"   ❌ PIOR QUE DAYTRADER: {eg_eq:.3f} < {dt_eq:.3f}")
        print(f"   ❌ ABAIXO DO ORIGINAL: {eg_eq:.3f} < {target_original}")
        final_verdict = "🔴 EXPERTGAIN V2 FALHOU"
    
    print(f"\n{final_verdict}")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    if expertgain_success:
        print(f"   ✅ ExpertGain V2 atingiu o objetivo!")
        print(f"   ✅ Pode ser usado em produção")
    elif improvement_over_daytrader:
        print(f"   🔄 Continuar treinamento para atingir {target_expertgain}")
        print(f"   📊 Ajustar hiperparâmetros se necessário")
    else:
        print(f"   🔄 Revisar estratégia de fine-tuning")
        print(f"   ⚠️ Possível overfitting ou configuração inadequada")

def main():
    """Função principal"""
    
    print("🔥 COMPARAÇÃO FINAL: EXPERTGAIN V2 vs DAYTRADER")
    print("=" * 60)
    print("🎯 Objetivo: Entry Quality 0.488 → 0.55+")
    print("📊 Teste: 3 runs x 5000 steps cada")
    print("💰 Portfolio: $500 inicial")
    print("=" * 60)
    
    # Testar ExpertGain V2
    expertgain_metrics = test_model(EXPERTGAIN_CHECKPOINT, "EXPERTGAIN V2")
    
    # Testar DayTrader
    daytrader_metrics = test_model(DAYTRADER_CHECKPOINT, "DAYTRADER FINAL")
    
    if expertgain_metrics and daytrader_metrics:
        # Comparação lado a lado
        compare_models(expertgain_metrics, daytrader_metrics)
    else:
        print("❌ Falha ao testar um ou ambos os modelos")

if __name__ == "__main__":
    print(f"🚀 INICIANDO COMPARAÇÃO FINAL - {datetime.now().strftime('%H:%M:%S')}")
    main()
    print(f"\n✅ COMPARAÇÃO CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")