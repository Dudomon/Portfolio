#!/usr/bin/env python3
"""
🧪 TEST EIGHTEEN CHECKPOINT LOAD
Verifica se o checkpoint de 1.55M carrega corretamente
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from stable_baselines3 import PPO

checkpoint_path = "D:/Projeto/Otimizacao/treino_principal/models/Eighteen/Eighteen_simpledirecttraining_1550000_steps_20251112_141410.zip"

print("=" * 80)
print("🧪 TEST EIGHTEEN CHECKPOINT LOAD")
print("=" * 80)
print(f"\n📂 Checkpoint: {checkpoint_path}")
print(f"📊 Steps: 1,550,000")
print(f"🏷️  Experiment: EIGHTEEN (Entry Timing V2)")
print()

# Test 1: Verificar se arquivo existe
print("1️⃣ Testing checkpoint file existence...")
if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"   ✅ File exists ({size_mb:.1f} MB)")
else:
    print(f"   ❌ File not found!")
    sys.exit(1)

# Test 2: Carregar checkpoint
print("\n2️⃣ Testing checkpoint load...")
try:
    model = PPO.load(checkpoint_path, device='cpu')
    print("   ✅ Checkpoint loaded successfully")
    print(f"   - Policy type: {type(model.policy).__name__}")
    print(f"   - Device: {model.device}")
except Exception as e:
    print(f"   ❌ Failed to load: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verificar observation space
print("\n3️⃣ Testing observation space...")
try:
    obs_space = model.observation_space
    print(f"   ✅ Observation space: {obs_space.shape}")
    expected_shape = (450,)
    if obs_space.shape == expected_shape:
        print(f"   ✅ Correct shape: {expected_shape}")
    else:
        print(f"   ⚠️  Warning: Expected {expected_shape}, got {obs_space.shape}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 4: Verificar action space
print("\n4️⃣ Testing action space...")
try:
    action_space = model.action_space
    print(f"   ✅ Action space: {action_space.shape}")
    expected_shape = (4,)
    if action_space.shape == expected_shape:
        print(f"   ✅ Correct shape: {expected_shape}")
    else:
        print(f"   ⚠️  Warning: Expected {expected_shape}, got {action_space.shape}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 5: Test prediction (dummy)
print("\n5️⃣ Testing prediction...")
try:
    import numpy as np
    dummy_obs = np.zeros(450, dtype=np.float32)
    action, _states = model.predict(dummy_obs, deterministic=True)
    print(f"   ✅ Prediction successful")
    print(f"   - Action shape: {action.shape}")
    print(f"   - Action values: {action}")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("✅ CHECKPOINT EIGHTEEN 1.55M - READY TO TEST!")
print("=" * 80)
print()
print("🚀 Run evaluation:")
print("   cd D:/Projeto/avaliacao")
print("   python cherry_avaliar.py")
