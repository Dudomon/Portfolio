"""
🧪 TESTE RÁPIDO - Validação das métricas de win rate corrigidas
Executa apenas 1 episódio para verificar se as métricas estão sendo calculadas corretamente
"""

import sys
import os
sys.path.append('D:\\Projeto')

import numpy as np
from cherry import TradingEnv
import stable_baselines3 as sb3
from stable_baselines3 import PPO
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)

def teste_metricas_rapido():
    """Teste rápido das métricas com apenas 1 episódio"""
    
    print("🧪 TESTE RÁPIDO - Validação das métricas de win rate")
    print("=" * 60)
    
    # Carregar dados (similar ao avaliar_cherry_optimized.py)
    print("📊 Carregando dados...")
    import pandas as pd
    data_path = "D:/Projeto/data/GC=F_YAHOO_20250821_161220.csv"
    data = pd.read_csv(data_path)
    print(f"   Dados carregados: {len(data)} barras")
    
    # Criar environment com configurações similares
    print("🏗️ Criando environment...")
    env = TradingEnv(
        df=data,
        window_size=20,
        is_training=True,
        initial_balance=500.0,
        trading_params={
            'min_lot_size': 0.02,
            'max_lot_size': 0.03,
            'enable_shorts': True,
            'max_positions': 2
        }
    )
    
    # Carregar um modelo específico
    model_path = "D:/Projeto/Modelo PPO Trader/checkpoints/Cherry_simpledirecttraining_400000_steps_20250907_230855.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    print(f"📋 Carregando modelo: {model_path}")
    model = PPO.load(model_path, env=env)
    
    # Executar 1 episódio
    print("🎯 Executando 1 episódio de teste...")
    
    obs = env.reset()
    step_count = 0
    episode_reward = 0
    done = False
    
    INITIAL_PORTFOLIO = 500.0
    
    while not done and step_count < 1000:  # Máximo 1000 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"   Step {step_count}: Portfolio ${env.portfolio_value:.2f}")
    
    # 🚨 COLETAR MÉTRICAS REAIS DO TRADING (lógica corrigida)
    portfolio_pnl = env.portfolio_value - INITIAL_PORTFOLIO
    trades_list = getattr(env, 'trades', [])
    
    print("\n📊 MÉTRICAS FINAIS:")
    print(f"   Portfolio inicial: ${INITIAL_PORTFOLIO:.2f}")
    print(f"   Portfolio final: ${env.portfolio_value:.2f}")
    print(f"   PnL real: ${portfolio_pnl:.2f}")
    print(f"   Episode reward: {episode_reward:.4f}")
    print(f"   Total steps: {step_count}")
    print(f"   Total trades: {len(trades_list)}")
    
    # Calcular win/loss de trades individuais
    if trades_list:
        winning_trades = sum(1 for trade in trades_list if trade.get('pnl_usd', 0) > 0)
        losing_trades = sum(1 for trade in trades_list if trade.get('pnl_usd', 0) < 0)
        total_trades_real = len(trades_list)
        
        # 🎯 WIN RATE CORRETO - baseado em trades individuais
        real_win_rate = (winning_trades / total_trades_real * 100) if total_trades_real > 0 else 0
        
        # WIN RATE ANTIGO (INCORRETO) - baseado em episode reward
        old_win_rate = 100 if episode_reward > 0 else 0  # Sempre daria 100% se reward > 0
        
        print(f"\n🎯 ANÁLISE TRADES INDIVIDUAIS:")
        print(f"   Trades vencedores: {winning_trades}")
        print(f"   Trades perdedores: {losing_trades}")
        print(f"   Win rate CORRETO: {real_win_rate:.1f}%")
        print(f"   Win rate ANTIGO (incorreto): {old_win_rate:.1f}%")
        
        # Mostrar alguns trades para validação
        print(f"\n📋 PRIMEIROS 3 TRADES (para validação):")
        for i, trade in enumerate(trades_list[:3]):
            pnl = trade.get('pnl_usd', 0)
            tipo = "WIN" if pnl > 0 else "LOSS"
            print(f"   Trade {i+1}: ${pnl:.2f} ({tipo})")
    
    else:
        print("   ⚠️ Nenhum trade executado no episódio")
    
    print("\n✅ Teste concluído!")

if __name__ == "__main__":
    teste_metricas_rapido()