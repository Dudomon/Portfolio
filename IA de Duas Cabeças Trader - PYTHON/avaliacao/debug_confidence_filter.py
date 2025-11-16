#!/usr/bin/env python3
"""
🔍 DEBUG DO FILTRO DE CONFIANÇA - VERIFICAR SE ESTÁ FUNCIONANDO
"""

import sys
sys.path.append("D:/Projeto")

from silus import TradingEnv
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
import torch

print("🔍 DEBUGGING FILTRO DE CONFIANÇA")
print("=" * 50)

# Carregar dataset pequeno para teste rápido
dataset_path = "D:/Projeto/data/GC=F_YAHOO_20250821_161220.csv"
df = pd.read_csv(dataset_path).head(1000)  # Só 1000 linhas para teste rápido

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

# Criar ambiente de teste
trading_params = {
    'base_lot_size': 0.02,
    'max_lot_size': 0.03,
    'initial_balance': 500.0,
    'target_trades_per_day': 18,
    'stop_loss_range': (2.0, 8.0),
    'take_profit_range': (3.0, 15.0)
}

print("📊 Criando TradingEnv para debug...")
env = TradingEnv(
    df,
    window_size=20,
    is_training=False,
    initial_balance=500.0,
    trading_params=trading_params
)

print("✅ Ambiente criado")

# Carregar modelo SILUS para testar
checkpoint_path = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_simpledirecttraining_2750000_steps_20250828_041024.zip"

print(f"🤖 Carregando modelo: {checkpoint_path}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    model = RecurrentPPO.load(checkpoint_path, device=device)
    print("✅ Modelo carregado")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    exit()

# Testar filtro original (0.8)
print(f"\n🧪 TESTE 1: FILTRO ORIGINAL (0.8)")

# Reset ambiente
env.current_step = 20
env.portfolio_value = 500.0
obs = env._get_observation()
lstm_states = None

confidence_rejections = 0
confidence_accepts = 0
total_actions = 0

for i in range(100):  # Testar 100 steps
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
    
    # Verificar confiança da ação
    if len(action) > 1:
        entry_decision = action[0]
        confidence = action[1]
        
        if entry_decision > 0.33:  # Se é uma ação de entrada (não HOLD)
            total_actions += 1
            if confidence < 0.8:
                confidence_rejections += 1
                print(f"   Step {i}: REJECT confidence={confidence:.3f} < 0.8")
            else:
                confidence_accepts += 1
                print(f"   Step {i}: ACCEPT confidence={confidence:.3f} >= 0.8")
    
    obs, reward, done, info = env.step(action)
    if done:
        break

print(f"\n📊 RESULTADOS FILTRO 0.8:")
print(f"   Total ações de entrada: {total_actions}")
print(f"   Rejeitadas (conf < 0.8): {confidence_rejections}")
print(f"   Aceitas (conf >= 0.8): {confidence_accepts}")
if total_actions > 0:
    print(f"   Taxa rejeição: {confidence_rejections/total_actions*100:.1f}%")

# Agora testar com filtro modificado (0.6)
print(f"\n🧪 TESTE 2: FILTRO MODIFICADO (0.6)")

# Reset ambiente
env.current_step = 20
env.portfolio_value = 500.0
obs = env._get_observation()
lstm_states = None

# Modificar filtro dinamicamente
import sys
silus_module = sys.modules['silus']
original_threshold = getattr(silus_module, 'MIN_CONFIDENCE_THRESHOLD', 0.8)
silus_module.MIN_CONFIDENCE_THRESHOLD = 0.6

print(f"🔧 Filtro alterado de {original_threshold} para {silus_module.MIN_CONFIDENCE_THRESHOLD}")

confidence_rejections_06 = 0
confidence_accepts_06 = 0
total_actions_06 = 0

for i in range(100):  # Testar 100 steps
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
    
    # Verificar confiança da ação
    if len(action) > 1:
        entry_decision = action[0]
        confidence = action[1]
        
        if entry_decision > 0.33:  # Se é uma ação de entrada (não HOLD)
            total_actions_06 += 1
            if confidence < 0.6:
                confidence_rejections_06 += 1
                print(f"   Step {i}: REJECT confidence={confidence:.3f} < 0.6")
            else:
                confidence_accepts_06 += 1
                print(f"   Step {i}: ACCEPT confidence={confidence:.3f} >= 0.6")
    
    obs, reward, done, info = env.step(action)
    if done:
        break

print(f"\n📊 RESULTADOS FILTRO 0.6:")
print(f"   Total ações de entrada: {total_actions_06}")
print(f"   Rejeitadas (conf < 0.6): {confidence_rejections_06}")
print(f"   Aceitas (conf >= 0.6): {confidence_accepts_06}")
if total_actions_06 > 0:
    print(f"   Taxa rejeição: {confidence_rejections_06/total_actions_06*100:.1f}%")

# Comparar
print(f"\n🔍 COMPARAÇÃO:")
print(f"   Filtro 0.8: {confidence_rejections}/{total_actions} rejeitadas")
print(f"   Filtro 0.6: {confidence_rejections_06}/{total_actions_06} rejeitadas")
print(f"   Diferença: {confidence_rejections - confidence_rejections_06} menos rejeições com filtro 0.6")

if confidence_rejections == confidence_rejections_06:
    print("⚠️ MESMO RESULTADO! Filtro pode não estar funcionando ou modelos já têm alta confiança")
else:
    print("✅ FILTRO FUNCIONANDO! Menos rejeições com threshold mais baixo")

# Restaurar filtro original
silus_module.MIN_CONFIDENCE_THRESHOLD = original_threshold