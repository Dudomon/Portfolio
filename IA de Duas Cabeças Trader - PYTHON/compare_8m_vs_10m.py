#!/usr/bin/env python3
"""
📊 COMPARAÇÃO 8M vs 10M STEPS - DAYTRADER V7
Testa checkpoint 8M e compara performance com 10M
"""

import sys
import os
import traceback
from datetime import datetime
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch

# Configuração específica - USAR OS MESMOS CHECKPOINTS QUE A AVALIAÇÃO ANTERIOR
CHECKPOINT_8M = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase4stresstesting_8000000_steps_20250811_195650.zip"
CHECKPOINT_10M = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase5integration_10000000_steps_20250812_003213.zip"  # MESMO USADO ANTES

INITIAL_PORTFOLIO = 500.0
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03
TEST_STEPS = 3000
NUM_EPISODES = 3
EPISODE_SPACING = 5000

def test_checkpoint(checkpoint_path, checkpoint_name):
    """🎯 Teste de um checkpoint específico"""
    
    print(f"\n{'=' * 80}")
    print(f"🎯 TESTANDO {checkpoint_name}")
    print(f"📂 {os.path.basename(checkpoint_path)}")
    print(f"{'=' * 80}")
    
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
        
        total_len = len(df)
        print(f"✅ Dataset carregado: {total_len:,} barras")
        
        # Parâmetros de trading
        trading_params = {
            'base_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'initial_balance': INITIAL_PORTFOLIO,
            'target_trades_per_day': 18,
            'stop_loss_range': (2.0, 8.0),
            'take_profit_range': (3.0, 15.0)
        }
        
        # Carregar modelo
        print(f"🤖 Carregando {checkpoint_name}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            model = RecurrentPPO.load(checkpoint_path, device=device)
            print("✅ Carregamento bem-sucedido")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return None
            
        model.policy.set_training_mode(False)
        print(f"✅ Modelo em modo inferência")
        
        # Executar testes
        print(f"🚀 Iniciando {NUM_EPISODES} episódios...")
        
        all_episodes = []
        total_returns = []
        
        for episode_num in range(NUM_EPISODES):
            print(f"\n🎮 EPISÓDIO {episode_num + 1}/{NUM_EPISODES}")
            
            # Selecionar pedaço do dataset
            start_idx = episode_num * EPISODE_SPACING
            if start_idx + TEST_STEPS + 100 > total_len:
                start_idx = total_len - TEST_STEPS - 100
            
            end_idx = start_idx + TEST_STEPS + 100
            episode_df = df.iloc[start_idx:end_idx].copy()
            
            print(f"📊 Período: {episode_df.index.min()} até {episode_df.index.max()}")
            
            # Criar ambiente
            env = TradingEnv(
                episode_df,
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
            
            trades_count = 0
            actions_log = []
            
            while not done and step < TEST_STEPS:
                # Predição em modo inferência
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                obs, reward, done, info = env.step(action)
                
                # Log da ação
                actions_log.append({
                    'step': step,
                    'entry_decision': int(action[0]),
                    'entry_quality': float(action[1]) if len(action) > 1 else 0.0,
                    'portfolio_value': env.portfolio_value
                })
                
                # Contar trades
                if hasattr(info, 'get') and info.get('trade_closed', False):
                    trades_count += 1
                    if trades_count <= 3:  # Log primeiros 3 trades
                        print(f"  💼 Trade #{trades_count}: {info.get('trade_type', 'unknown')} PnL=${info.get('trade_pnl', 0):.2f}")
                
                if (step + 1) % 1500 == 0:
                    print(f"  Step {step+1}/{TEST_STEPS} - Portfolio: ${env.portfolio_value:.2f} - Trades: {trades_count}")
                
                step += 1
            
            # Análise do episódio
            final_portfolio = env.portfolio_value
            episode_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
            
            episode_result = {
                'episode': episode_num + 1,
                'period': f"{episode_df.index.min()} até {episode_df.index.max()}",
                'initial_portfolio': INITIAL_PORTFOLIO,
                'final_portfolio': final_portfolio,
                'return_pct': episode_return,
                'trades_count': trades_count,
                'actions_log': actions_log
            }
            
            all_episodes.append(episode_result)
            total_returns.append(episode_return)
            
            print(f"✅ Resultado: ${INITIAL_PORTFOLIO:.2f} → ${final_portfolio:.2f} ({episode_return:+.2f}%) - {trades_count} trades")
            
            # Limpeza
            del env
            del episode_df
        
        # Análise final
        avg_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        min_return = min(total_returns)
        max_return = max(total_returns)
        positive_episodes = len([r for r in total_returns if r > 0])
        total_trades = sum(ep['trades_count'] for ep in all_episodes)
        
        # Análise de ações do último episódio
        last_actions = all_episodes[-1]['actions_log'] if all_episodes else []
        if last_actions:
            entry_decisions = [a['entry_decision'] for a in last_actions]
            entry_qualities = [a['entry_quality'] for a in last_actions]
            
            hold_pct = (sum(1 for d in entry_decisions if d == 0) / len(entry_decisions)) * 100
            long_pct = (sum(1 for d in entry_decisions if d == 1) / len(entry_decisions)) * 100
            short_pct = (sum(1 for d in entry_decisions if d == 2) / len(entry_decisions)) * 100
            avg_quality = np.mean(entry_qualities)
        else:
            hold_pct = long_pct = short_pct = avg_quality = 0
        
        print(f"\n📊 RESUMO {checkpoint_name}:")
        print(f"💵 Retorno Médio: {avg_return:+.2f}% (σ={std_return:.2f}%)")
        print(f"📈 Range: {min_return:+.2f}% até {max_return:+.2f}%")
        print(f"🎯 Episódios Lucrativos: {positive_episodes}/{NUM_EPISODES} ({(positive_episodes/NUM_EPISODES)*100:.1f}%)")
        print(f"📊 Total Trades: {total_trades}")
        print(f"🎮 Ações: HOLD={hold_pct:.1f}%, LONG={long_pct:.1f}%, SHORT={short_pct:.1f}%")
        print(f"⭐ Entry Quality: {avg_quality:.3f}")
        
        # Limpeza de memória
        del model
        torch.cuda.empty_cache()
        
        return {
            'checkpoint_name': checkpoint_name,
            'avg_return': avg_return,
            'std_return': std_return,
            'min_return': min_return,
            'max_return': max_return,
            'positive_episodes': positive_episodes,
            'total_episodes': NUM_EPISODES,
            'total_trades': total_trades,
            'hold_pct': hold_pct,
            'long_pct': long_pct,
            'short_pct': short_pct,
            'avg_quality': avg_quality,
            'all_episodes': all_episodes
        }
        
    except Exception as e:
        print(f\"❌ ERRO: {e}\")
        print(f\"Detalhes: {traceback.format_exc()}\")
        return None

def main():
    print("🔥 COMPARAÇÃO DAYTRADER: 8M vs 10M STEPS")
    print("=" * 80)
    print(f"💰 Portfolio Inicial: ${INITIAL_PORTFOLIO}")
    print(f"📊 Episódios: {NUM_EPISODES} × {TEST_STEPS} steps")
    print(f"🎯 Modo: Inferência não-determinística")
    print("=" * 80)
    
    # Testar checkpoint 8M
    result_8m = test_checkpoint(CHECKPOINT_8M, "8M STEPS (Phase 4 - Stress Testing)")
    
    # Testar checkpoint 10M  
    result_10m = test_checkpoint(CHECKPOINT_10M, "10M STEPS (Phase 5 - Integration)")
    
    # Comparação final
    if result_8m and result_10m:
        print(\"\\n\" + \"=\" * 100)
        print(\"🏆 COMPARAÇÃO FINAL: 8M vs 10M\")
        print(\"=\" * 100)
        
        print(f\"📈 RETORNO MÉDIO:\")
        print(f\"   8M:  {result_8m['avg_return']:+7.2f}% (σ={result_8m['std_return']:.2f}%)\")
        print(f\"   10M: {result_10m['avg_return']:+7.2f}% (σ={result_10m['std_return']:.2f}%)\")
        
        delta_return = result_8m['avg_return'] - result_10m['avg_return']
        print(f\"   📊 Diferença: {delta_return:+.2f}% ({'✅ 8M melhor' if delta_return > 0 else '❌ 10M melhor'})\")
        
        print(f\"\\n🎯 EPISÓDIOS LUCRATIVOS:\")
        print(f\"   8M:  {result_8m['positive_episodes']}/{result_8m['total_episodes']} ({(result_8m['positive_episodes']/result_8m['total_episodes'])*100:.1f}%)\")
        print(f\"   10M: {result_10m['positive_episodes']}/{result_10m['total_episodes']} ({(result_10m['positive_episodes']/result_10m['total_episodes'])*100:.1f}%)\")
        
        print(f\"\\n📊 ATIVIDADE DE TRADING:\")
        print(f\"   8M:  {result_8m['total_trades']} trades totais\")
        print(f\"   10M: {result_10m['total_trades']} trades totais\")
        
        print(f\"\\n🎮 COMPORTAMENTO (Entry Decisions):\")
        print(f\"   8M:  HOLD={result_8m['hold_pct']:.1f}%, LONG={result_8m['long_pct']:.1f}%, SHORT={result_8m['short_pct']:.1f}%\")
        print(f\"   10M: HOLD={result_10m['hold_pct']:.1f}%, LONG={result_10m['long_pct']:.1f}%, SHORT={result_10m['short_pct']:.1f}%\")
        
        # Determinação do vencedor
        score_8m = 0
        score_10m = 0
        
        # Critério 1: Retorno médio
        if result_8m['avg_return'] > result_10m['avg_return']:
            score_8m += 1
        else:
            score_10m += 1
            
        # Critério 2: Consistência (episódios lucrativos)
        if result_8m['positive_episodes'] > result_10m['positive_episodes']:
            score_8m += 1
        else:
            score_10m += 1
            
        # Critério 3: Atividade (trading activity)
        if result_8m['total_trades'] > result_10m['total_trades']:
            score_8m += 1
        else:
            score_10m += 1
        
        print(f\"\\n🏆 VEREDITO FINAL:\")
        print(f\"   Score 8M:  {score_8m}/3\")
        print(f\"   Score 10M: {score_10m}/3\")
        
        if score_8m > score_10m:
            print(\"   🥇 VENCEDOR: 8M STEPS (menos overtraining)\")
            print(\"   💡 RECOMENDAÇÃO: Usar checkpoint 8M para produção\")
        elif score_10m > score_8m:
            print(\"   🥇 VENCEDOR: 10M STEPS (mais treinamento)\")
            print(\"   💡 RECOMENDAÇÃO: Usar checkpoint 10M para produção\")
        else:
            print(\"   🤝 EMPATE - ambos equivalentes\")
            print(\"   💡 RECOMENDAÇÃO: Usar 8M (menos risco de overtraining)\")
    
    print(f\"\\n✅ COMPARAÇÃO CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}\")

if __name__ == \"__main__\":
    main()