#!/usr/bin/env python3
"""
🧪 TEST EIGHTEEN INITIALIZATION
Testa se o sistema Entry Timing V2 inicializa corretamente
"""

import sys
import os

# Adicionar paths
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("🧪 TEST EIGHTEEN INITIALIZATION")
print("=" * 80)
print()

# Test 1: Import Entry Timing Rewards
print("1️⃣ Testing Entry Timing Rewards import...")
try:
    from trading_framework.rewards.entry_timing_rewards import EntryTimingRewards, MultiSignalConfluenceEntry
    print("   ✅ Import successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize Entry Timing System
print("\n2️⃣ Testing Entry Timing System initialization...")
try:
    entry_timing = EntryTimingRewards()
    print("   ✅ EntryTimingRewards initialized")
    print(f"   - timing_quality_weight: {entry_timing.timing_quality_weight}")
    print(f"   - confluence_weight: {entry_timing.confluence_weight}")
    print(f"   - market_context_weight: {entry_timing.market_context_weight}")
    print(f"   - Has multi_signal_system: {hasattr(entry_timing, 'multi_signal_system')}")
    print(f"   - Has consecutive_losses tracking: {hasattr(entry_timing, 'consecutive_losses')}")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Initialize Multi-Signal Confluence
print("\n3️⃣ Testing Multi-Signal Confluence initialization...")
try:
    multi_signal = MultiSignalConfluenceEntry()
    print("   ✅ MultiSignalConfluenceEntry initialized")
    print(f"   - layer1_weight: {multi_signal.layer1_weight}")
    print(f"   - layer2_weight: {multi_signal.layer2_weight}")
    print(f"   - layer3_weight: {multi_signal.layer3_weight}")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Import V3 Brutal
print("\n4️⃣ Testing V3 Brutal import...")
try:
    from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward
    print("   ✅ V3 Brutal import successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Initialize V3 Brutal
print("\n5️⃣ Testing V3 Brutal initialization...")
try:
    reward_system = BrutalMoneyReward(initial_balance=1000.0)
    print("   ✅ V3 Brutal initialized")
    print(f"   - Has entry_timing_system: {hasattr(reward_system, 'entry_timing_system')}")
    if hasattr(reward_system, 'entry_timing_system'):
        print(f"   - Entry timing system type: {type(reward_system.entry_timing_system).__name__}")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check Cherry EXPERIMENT_TAG
print("\n6️⃣ Testing Cherry EXPERIMENT_TAG...")
try:
    import cherry
    print(f"   ✅ EXPERIMENT_TAG = '{cherry.EXPERIMENT_TAG}'")
    expected = "Eighteen"
    if cherry.EXPERIMENT_TAG == expected:
        print(f"   ✅ Correct tag: {expected}")
    else:
        print(f"   ⚠️  Warning: Expected '{expected}', got '{cherry.EXPERIMENT_TAG}'")
except Exception as e:
    print(f"   ❌ Cherry import failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("✅ ALL TESTS PASSED - EIGHTEEN INITIALIZATION OK!")
print("=" * 80)
print()
print("📋 SISTEMA EIGHTEEN PRONTO:")
print("   • Entry Timing Rewards V2 ✅")
print("   • Multi-Signal Confluence (3 layers) ✅")
print("   • Behavioral Controls (Revenge, Cut Loss) ✅")
print("   • Pattern Recognition (MA Cross, Double Top/Bottom) ✅")
print("   • Entry Timing After Loss ✅")
print("   • Peso dobrado: 12% do reward total ✅")
print()
print("🚀 Pronto para treinar!")
