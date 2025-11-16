#!/usr/bin/env python3
"""
🚨🚨🚨 FIX EMERGENCIAL IMEDIATO V8HERITAGE 🚨🚨🚨
APLICAR NO MODELO QUE ESTÁ RODANDO AGORA!

weight_hh_l0: 100% zeros - LSTMs MORTOS
entry_quality_head: 70% zeros - DECISION MAKING MORTO
"""

import torch
import torch.nn as nn
import numpy as np

def emergency_fix_v8heritage_now():
    """
    🚨 APLIQUE ESTE FIX NO SEU MODELO AGORA MESMO!
    
    INSTRUÇÕES:
    1. Salve este arquivo como emergency_fix_v8.py
    2. No seu script daytrader8dim, adicione:
    
    from emergency_fix_v8 import emergency_fix_v8heritage_now
    
    # APÓS carregar o modelo:
    emergency_fix_v8heritage_now(model)
    
    3. Continue o treinamento
    """
    
    def apply_emergency_fix(model):
        print("\n" + "🚨" * 30)
        print("🚨 APLICANDO FIX EMERGENCIAL V8HERITAGE")
        print("🚨" * 30)
        
        if not hasattr(model, 'policy'):
            print("❌ ERRO: Modelo não tem policy")
            return False
            
        policy = model.policy
        
        if policy.__class__.__name__ != 'TwoHeadV8Heritage':
            print("❌ ERRO: Não é TwoHeadV8Heritage")
            return False
            
        fixes_applied = []
        
        # ==============================================
        # 🔥 FIX 1: LSTMs MORTOS (100% zeros)
        # ==============================================
        
        if hasattr(policy, 'neural_architecture'):
            neural_arch = policy.neural_architecture
            
            # Fix Actor LSTM
            if hasattr(neural_arch, 'actor_lstm'):
                print("🔧 FIXING Actor LSTM (100% zeros)...")
                lstm = neural_arch.actor_lstm
                
                for name, param in lstm.named_parameters():
                    if 'weight_hh' in name:
                        with torch.no_grad():
                            # FORÇA inicialização orthogonal
                            nn.init.orthogonal_(param.data, gain=1.0)
                            
                            # Verificação
                            zeros_before = (param.data.abs() < 1e-8).float().mean().item()
                            if zeros_before < 0.05:  # <5% zeros é bom
                                print(f"   ✅ {name}: {zeros_before*100:.1f}% zeros - FIXED!")
                                fixes_applied.append(f"actor_lstm.{name}")
                            else:
                                print(f"   ⚠️ {name}: {zeros_before*100:.1f}% zeros - STILL HIGH")
                    
                    elif 'bias' in name and param.size(0) >= 4:
                        with torch.no_grad():
                            # FORÇAR mudança nos bias
                            param.data.zero_()
                            # Forget gate = 1.0 (segunda porta dos 4 gates)
                            n = param.size(0)
                            start_idx = n // 4
                            end_idx = n // 2
                            param.data[start_idx:end_idx] = 1.0
                            
                            # Verificar se aplicou
                            zeros_after = (param.data.abs() < 1e-8).float().mean().item()
                            print(f"   ✅ {name}: {zeros_after*100:.1f}% zeros, forget_gate=1.0")
                            fixes_applied.append(f"actor_lstm.{name}")
            
            # Fix Critic LSTM  
            if hasattr(neural_arch, 'critic_lstm'):
                print("🔧 FIXING Critic LSTM (100% zeros)...")
                lstm = neural_arch.critic_lstm
                
                for name, param in lstm.named_parameters():
                    if 'weight_hh' in name:
                        with torch.no_grad():
                            nn.init.orthogonal_(param.data, gain=1.0)
                            
                            zeros_before = (param.data.abs() < 1e-8).float().mean().item()
                            if zeros_before < 0.05:
                                print(f"   ✅ {name}: {zeros_before*100:.1f}% zeros - FIXED!")
                                fixes_applied.append(f"critic_lstm.{name}")
                            else:
                                print(f"   ⚠️ {name}: {zeros_before*100:.1f}% zeros - STILL HIGH")
                    
                    elif 'bias' in name and param.size(0) >= 4:
                        with torch.no_grad():
                            # FORÇAR mudança nos bias
                            param.data.zero_()
                            n = param.size(0)
                            start_idx = n // 4  
                            end_idx = n // 2
                            param.data[start_idx:end_idx] = 1.0
                            
                            # Verificar se aplicou
                            zeros_after = (param.data.abs() < 1e-8).float().mean().item()
                            print(f"   ✅ {name}: {zeros_after*100:.1f}% zeros, forget_gate=1.0")
                            fixes_applied.append(f"critic_lstm.{name}")
        
        # ==============================================
        # 🔥 FIX 2: DecisionMaker MORTO (70% zeros)  
        # ==============================================
        
        if hasattr(policy, 'decision_maker'):
            print("🔧 FIXING DecisionMaker (70% zeros)...")
            decision_maker = policy.decision_maker
            
            # Fix entry_quality_head
            if hasattr(decision_maker, 'entry_quality_head'):
                print("   🎯 Fixing entry_quality_head...")
                head = decision_maker.entry_quality_head
                
                for name, module in head.named_modules():
                    if isinstance(module, nn.Linear):
                        with torch.no_grad():
                            # Re-inicializar completamente - FORÇAR mudança
                            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                            if module.bias is not None:
                                # FORÇAR bias para valores pequenos mas não zero
                                module.bias.data.uniform_(-0.01, 0.01)
                            
                            # Verificar
                            weight_zeros = (module.weight.abs() < 1e-8).float().mean().item()
                            if weight_zeros < 0.15:  # <15% é aceitável
                                print(f"      ✅ {name}: {weight_zeros*100:.1f}% zeros - FIXED!")
                                fixes_applied.append(f"entry_quality_head.{name}")
                            else:
                                print(f"      ⚠️ {name}: {weight_zeros*100:.1f}% zeros - STILL HIGH")
            
            # Fix position_management_head
            if hasattr(decision_maker, 'position_management_head'):
                print("   📊 Fixing position_management_head...")
                head = decision_maker.position_management_head
                
                for name, module in head.named_modules():
                    if isinstance(module, nn.Linear):
                        with torch.no_grad():
                            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                            if module.bias is not None:
                                # FORÇAR bias para valores pequenos mas não zero
                                module.bias.data.uniform_(-0.01, 0.01)
                            
                            weight_zeros = (module.weight.abs() < 1e-8).float().mean().item()
                            if weight_zeros < 0.15:
                                print(f"      ✅ {name}: {weight_zeros*100:.1f}% zeros - FIXED!")
                                fixes_applied.append(f"position_management_head.{name}")
            
            # Fix risk_weighting
            if hasattr(decision_maker, 'risk_weighting'):
                print("   ⚖️ Fixing risk_weighting...")
                head = decision_maker.risk_weighting
                
                for name, module in head.named_modules():
                    if isinstance(module, nn.Linear):
                        with torch.no_grad():
                            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                            if module.bias is not None:
                                # FORÇAR bias para valores pequenos mas não zero
                                module.bias.data.uniform_(-0.01, 0.01)
                            
                            weight_zeros = (module.weight.abs() < 1e-8).float().mean().item()
                            if weight_zeros < 0.15:
                                print(f"      ✅ {name}: {weight_zeros*100:.1f}% zeros - FIXED!")
                                fixes_applied.append(f"risk_weighting.{name}")
        
        # ==============================================
        # 🔥 FIX 3: Transformer Attention (33% zeros)
        # ==============================================
        
        if hasattr(policy, 'features_extractor'):
            print("🔧 FIXING Transformer Attention (33% zeros)...")
            transformer = policy.features_extractor
            
            for name, param in transformer.named_parameters():
                if 'self_attn.in_proj_bias' in name:
                    with torch.no_grad():
                        # FORÇAR bias para valores pequenos mas não zero
                        param.data.uniform_(-0.01, 0.01)
                        
                        zeros_after = (param.data.abs() < 1e-8).float().mean().item()
                        print(f"   ✅ {name}: {zeros_after*100:.1f}% zeros - FIXED!")
                        fixes_applied.append(f"transformer.{name}")
        
        # ==============================================
        # RELATÓRIO FINAL
        # ==============================================
        
        print("\n" + "✅" * 30)
        print("✅ FIX EMERGENCIAL COMPLETADO")
        print("✅" * 30)
        print(f"📊 Total de componentes corrigidos: {len(fixes_applied)}")
        
        if len(fixes_applied) > 0:
            print("🎯 Componentes corrigidos:")
            for fix in fixes_applied:
                print(f"   ✅ {fix}")
                
            print("\n🚀 MODELO DEVE ESTAR FUNCIONAL AGORA!")
            print("🔄 Continue o treinamento para ver os resultados")
            return True
        else:
            print("❌ NENHUM COMPONENTE FOI CORRIGIDO")
            print("🔍 Verifique se o modelo está correto")
            return False
    
    return apply_emergency_fix

# ==============================================
# 🚀 MODO DE USO IMEDIATO
# ==============================================

def apply_fix_now(model):
    """Aplique este fix AGORA no seu modelo"""
    fix_function = emergency_fix_v8heritage_now()
    return fix_function(model)

if __name__ == "__main__":
    print("🚨 EMERGENCY FIX V8HERITAGE - READY TO USE")
    print("IMPORT: from emergency_fix_v8 import apply_fix_now")
    print("USE: apply_fix_now(your_model)")