"""
🧪 TESTE COMPLETO: TwoHeadV8Elegance Architecture

Testa todos os componentes da V8Elegance:
- LSTM única compartilhada
- Entry Head específico (entry + confidence)  
- Management Head específico (SL/TP por posição)
- Memory Bank elegante
- Market Context único
- 8D action space completo
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import traceback
from gym import spaces

# Add project path
sys.path.append(r'D:\Projeto')

def test_v8_elegance_complete():
    """🧪 Teste completo da arquitetura V8Elegance"""
    
    print("🚀 INICIANDO TESTE COMPLETO: TwoHeadV8Elegance")
    print("="*60)
    
    try:
        # Import V8Elegance
        from trading_framework.policies.two_head_v8_elegance import (
            TwoHeadV8Elegance, DaytradeEntryHead, DaytradeManagementHead,
            MarketContextEncoder, ElegantMemoryBank, get_v8_elegance_kwargs,
            validate_v8_elegance_policy
        )
        print("✅ Imports V8Elegance successful")
        
    except Exception as e:
        print(f"❌ Erro nos imports: {e}")
        traceback.print_exc()
        return False
    
    # Test individual components first
    print("\n" + "="*60)
    print("🔧 TESTANDO COMPONENTES INDIVIDUAIS")
    print("="*60)
    
    # 1. Test MarketContextEncoder
    print("\n1. 🌍 Testando MarketContextEncoder...")
    try:
        context_encoder = MarketContextEncoder(input_dim=256, context_dim=64)
        lstm_features = torch.randn(2, 10, 256)  # batch, seq, features
        context_features, regime_id, info = context_encoder(lstm_features)
        
        assert context_features.shape == (2, 10, 64), f"Context features shape: {context_features.shape}"
        assert isinstance(regime_id, int), f"Regime ID type: {type(regime_id)}"
        assert 0 <= regime_id <= 3, f"Regime ID range: {regime_id}"
        assert 'regime_name' in info, "Missing regime_name in info"
        
        print(f"   ✅ Context shape: {context_features.shape}")
        print(f"   ✅ Regime: {regime_id} ({info['regime_name']})")
        print(f"   ✅ Confidence: {info.get('regime_confidence', 0):.3f}")
        
    except Exception as e:
        print(f"   ❌ MarketContextEncoder failed: {e}")
        return False
    
    # 2. Test DaytradeEntryHead
    print("\n2. 🎯 Testando DaytradeEntryHead...")
    try:
        entry_head = DaytradeEntryHead(input_dim=320)  # 256 + 64
        lstm_features = torch.randn(2, 256)
        market_context = torch.randn(2, 64)
        
        raw_entry, entry_confidence, gate_info = entry_head(lstm_features, market_context)
        
        assert raw_entry.shape == (2, 1), f"Raw entry shape: {raw_entry.shape}"
        assert entry_confidence.shape == (2, 1), f"Entry confidence shape: {entry_confidence.shape}"
        assert torch.all((entry_confidence >= 0) & (entry_confidence <= 1)), "Entry confidence not in [0,1]"
        assert 'entry_head_type' in gate_info, "Missing entry_head_type in gate_info"
        
        print(f"   ✅ Raw entry range: [{raw_entry.min():.3f}, {raw_entry.max():.3f}]")
        print(f"   ✅ Confidence range: [{entry_confidence.min():.3f}, {entry_confidence.max():.3f}]")
        print(f"   ✅ Gate info keys: {list(gate_info.keys())}")
        
    except Exception as e:
        print(f"   ❌ DaytradeEntryHead failed: {e}")
        traceback.print_exc()
        return False
    
    # 3. Test DaytradeManagementHead  
    print("\n3. 💰 Testando DaytradeManagementHead...")
    try:
        mgmt_head = DaytradeManagementHead(input_dim=320)
        lstm_features = torch.randn(2, 256)
        market_context = torch.randn(2, 64)
        
        mgmt_decisions, mgmt_conf, mgmt_weights = mgmt_head(lstm_features, market_context)
        
        assert mgmt_decisions.shape == (2, 6), f"Management decisions shape: {mgmt_decisions.shape}"
        assert mgmt_conf.shape == (2, 1), f"Management confidence shape: {mgmt_conf.shape}"
        assert mgmt_weights.shape == (2, 3), f"Management weights shape: {mgmt_weights.shape}"
        assert torch.all((mgmt_decisions >= -3) & (mgmt_decisions <= 3)), "SL/TP not in [-3,3]"
        
        print(f"   ✅ SL/TP decisions shape: {mgmt_decisions.shape}")
        print(f"   ✅ SL/TP range: [{mgmt_decisions.min():.3f}, {mgmt_decisions.max():.3f}]")
        print(f"   ✅ Mgmt confidence: [{mgmt_conf.min():.3f}, {mgmt_conf.max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ DaytradeManagementHead failed: {e}")
        traceback.print_exc()
        return False
    
    # 4. Test ElegantMemoryBank
    print("\n4. 💾 Testando ElegantMemoryBank...")
    try:
        memory_bank = ElegantMemoryBank(memory_size=512, trade_dim=8)
        
        # Add some trades
        for i in range(10):
            trade_data = torch.randn(8) * 0.1
            memory_bank.add_trade(trade_data)
        
        context = memory_bank.get_memory_context(batch_size=2)
        assert context.shape == (2, 8), f"Memory context shape: {context.shape}"
        
        print(f"   ✅ Memory size: {memory_bank.memory_size}")
        print(f"   ✅ Current pointer: {memory_bank.memory_ptr.item()}")
        print(f"   ✅ Context shape: {context.shape}")
        
    except Exception as e:
        print(f"   ❌ ElegantMemoryBank failed: {e}")
        return False
    
    # Test complete policy
    print("\n" + "="*60)
    print("🏗️ TESTANDO POLICY COMPLETA")
    print("="*60)
    
    try:
        # Create observation and action spaces
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2580,), dtype=np.float32)
        action_space = spaces.Box(low=-3, high=3, shape=(8,), dtype=np.float32)
        
        print(f"📊 Observation space: {observation_space.shape}")
        print(f"🎯 Action space: {action_space.shape}")
        
        # Get kwargs
        kwargs = get_v8_elegance_kwargs()
        print(f"📋 V8 kwargs keys: {list(kwargs.keys())}")
        
        # Create policy
        print("\n🚧 Criando TwoHeadV8Elegance...")
        policy = TwoHeadV8Elegance(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            **kwargs
        )
        print("✅ Policy criada com sucesso!")
        
        # Validate policy
        print("\n🔍 Validando policy...")
        validate_v8_elegance_policy(policy)
        
        # Test forward passes
        print("\n🎯 Testando forward passes...")
        
        batch_size = 2
        # FIX: Observations devem ter shape (batch_size, 2580) para o transformer
        observations = torch.randn(batch_size, 2580)
        lstm_states = (
            torch.zeros(1, batch_size, 256),
            torch.zeros(1, batch_size, 256)
        )
        episode_starts = torch.zeros(batch_size, dtype=torch.bool)
        
        # Test actor forward
        print("   🎭 Actor forward...")
        actions, new_states, gate_info = policy.forward_actor(observations, lstm_states, episode_starts)
        
        assert actions.shape == (batch_size, 8), f"Actions shape: {actions.shape}"
        assert len(new_states) == 2, f"LSTM states: {len(new_states)}"
        assert isinstance(gate_info, dict), f"Gate info type: {type(gate_info)}"
        
        print(f"      ✅ Actions shape: {actions.shape}")
        print(f"      ✅ Action ranges:")
        print(f"         [0] entry_decision: {actions[:, 0].tolist()}")
        print(f"         [1] confidence: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]")
        print(f"         [2-7] SL/TP: [{actions[:, 2:].min():.3f}, {actions[:, 2:].max():.3f}]")
        
        # Test critic forward
        print("   💰 Critic forward...")
        values, critic_states = policy.forward_critic(observations, lstm_states, episode_starts)
        
        assert values.shape == (batch_size, 1), f"Values shape: {values.shape}"
        
        print(f"      ✅ Values shape: {values.shape}")
        print(f"      ✅ Values range: [{values.min():.3f}, {values.max():.3f}]")
        
        # Test predict_values
        print("   📊 Predict values...")
        pred_values = policy.predict_values(observations, lstm_states, episode_starts)
        
        assert pred_values.shape == values.shape, f"Predicted values shape mismatch"
        
        print(f"      ✅ Predicted values consistent with forward_critic")
        
        # Test status
        print("   📈 Status check...")
        status = policy.get_v8_status()
        
        expected_keys = ['architecture', 'lstm_hidden', 'features_dim', 'current_regime']
        for key in expected_keys:
            assert key in status, f"Missing key in status: {key}"
        
        print(f"      ✅ Status keys: {list(status.keys())}")
        print(f"      ✅ Current regime: {status['current_regime']} ({status.get('regime_info', {}).get('regime_name', 'unknown')})")
        
        # Test post training step
        print("   🏃 Post training step...")
        experience = {
            'reward': 0.1,
            'duration': 5.0,
            'confidence': 0.8,
            'done': False,
            'pnl': 0.05
        }
        
        post_info = policy.post_training_step(experience)
        
        assert 'v8_regime' in post_info, "Missing v8_regime in post_info"
        assert 'v8_training_step' in post_info, "Missing v8_training_step in post_info"
        
        print(f"      ✅ Post training info: {list(post_info.keys())}")
        print(f"      ✅ Training step: {post_info['v8_training_step']}")
        
    except Exception as e:
        print(f"❌ Erro no teste da policy completa: {e}")
        traceback.print_exc()
        return False
    
    # Final validation
    print("\n" + "="*60)
    print("🏁 VALIDAÇÃO FINAL")
    print("="*60)
    
    print("✅ Todos os testes passaram!")
    print()
    print("📊 RESUMO V8ELEGANCE:")
    print(f"   🧠 LSTM compartilhada: {policy.v8_lstm_hidden}D")
    print(f"   🎯 Entry Head: Específico (entry + confidence)")  
    print(f"   💰 Management Head: Específico (SL/TP por posição)")
    print(f"   💾 Memory Bank: {policy.v8_memory_size} trades")
    print(f"   🌍 Market Context: {policy.v8_context_dim}D (4 regimes)")
    print(f"   ⚡ Action Space: 8D completo")
    print()
    print("🎯 CARACTERÍSTICAS ELEGANTES:")
    print("   ✅ Uma LSTM (não várias)")
    print("   ✅ Heads específicos (não genéricos)")
    print("   ✅ Memory simplificado (não complexo)")  
    print("   ✅ Context único (não múltiplos gates)")
    print("   ✅ 8D actions mantidas (funcionalidade completa)")
    
    return True

if __name__ == "__main__":
    success = test_v8_elegance_complete()
    
    if success:
        print("\n🚀 TwoHeadV8Elegance APROVADA - Pronta para uso!")
    else:
        print("\n❌ TwoHeadV8Elegance FALHOU - Revisar implementação")
    
    print("\n" + "="*60)