#!/usr/bin/env python3
"""
🚨 FIX EMERGENCIAL V8Heritage - ZEROS CRÍTICOS
Corrige inicialização de LSTMs que estão com 100% zeros nos weight_hh_l0
"""

import torch
import torch.nn as nn

def fix_v8_heritage_lstm_initialization():
    """🔧 Fix específico para LSTMs V8Heritage com zeros críticos"""
    
    print("🚨 APLICANDO FIX EMERGENCIAL PARA V8HERITAGE ZEROS")
    print("="*60)
    
    # Este fix deve ser aplicado APÓS criar o modelo V8Heritage
    
    def apply_robust_lstm_init(model):
        """Aplicar inicialização robusta nos LSTMs"""
        
        if not hasattr(model, 'policy'):
            print("❌ Modelo não tem atributo 'policy'")
            return False
            
        policy = model.policy
        
        if not hasattr(policy, 'neural_architecture'):
            print("❌ Policy não tem 'neural_architecture'") 
            return False
            
        neural_arch = policy.neural_architecture
        
        # Fix Actor LSTM
        if hasattr(neural_arch, 'actor_lstm'):
            print("🔧 Aplicando fix no Actor LSTM...")
            lstm = neural_arch.actor_lstm
            
            for name, param in lstm.named_parameters():
                if 'weight_hh' in name:
                    print(f"   🎯 CRITICAL FIX: {name}")
                    # FORÇA inicialização orthogonal com gain adequado
                    with torch.no_grad():
                        nn.init.orthogonal_(param.data, gain=1.0)
                        print(f"      ✅ Orthogonal init aplicado (gain=1.0)")
                        
                        # Verificar se não há zeros extremos
                        zero_count = (param.data.abs() < 1e-8).sum().item()
                        total_params = param.data.numel()
                        zero_ratio = zero_count / total_params
                        print(f"      📊 Zeros após fix: {zero_ratio*100:.1f}%")
                        
                elif 'bias' in name and param.size(0) >= 4:
                    print(f"   🔧 Fixing bias: {name}")
                    with torch.no_grad():
                        param.data.fill_(0.0)
                        # Forget gate bias = 1.0
                        n = param.size(0)
                        forget_start = n // 4
                        forget_end = n // 2
                        param.data[forget_start:forget_end].fill_(1.0)
                        print(f"      ✅ Forget gate bias = 1.0")
        
        # Fix Critic LSTM
        if hasattr(neural_arch, 'critic_lstm'):
            print("🔧 Aplicando fix no Critic LSTM...")
            lstm = neural_arch.critic_lstm
            
            for name, param in lstm.named_parameters():
                if 'weight_hh' in name:
                    print(f"   🎯 CRITICAL FIX: {name}")
                    with torch.no_grad():
                        nn.init.orthogonal_(param.data, gain=1.0)
                        print(f"      ✅ Orthogonal init aplicado (gain=1.0)")
                        
                        zero_count = (param.data.abs() < 1e-8).sum().item()
                        total_params = param.data.numel()
                        zero_ratio = zero_count / total_params
                        print(f"      📊 Zeros após fix: {zero_ratio*100:.1f}%")
                        
                elif 'bias' in name and param.size(0) >= 4:
                    print(f"   🔧 Fixing bias: {name}")
                    with torch.no_grad():
                        param.data.fill_(0.0)
                        n = param.size(0)
                        forget_start = n // 4
                        forget_end = n // 2
                        param.data[forget_start:forget_end].fill_(1.0)
                        print(f"      ✅ Forget gate bias = 1.0")
        
        print("✅ Fix V8Heritage LSTMs aplicado com sucesso!")
        return True
    
    return apply_robust_lstm_init

def diagnose_v8_heritage_zeros(model):
    """Diagnóstico específico dos zeros V8Heritage"""
    
    print("\n🔍 DIAGNÓSTICO V8HERITAGE ZEROS")
    print("="*50)
    
    if not hasattr(model, 'policy'):
        print("❌ Modelo inválido")
        return
    
    policy = model.policy
    policy_class = policy.__class__.__name__
    
    print(f"🏗️ Policy: {policy_class}")
    
    if policy_class != 'TwoHeadV8Heritage':
        print("⚠️ Não é V8Heritage, diagnóstico pode ser impreciso")
        return
    
    critical_components = []
    
    # Verificar neural architecture
    if hasattr(policy, 'neural_architecture'):
        neural_arch = policy.neural_architecture
        
        for lstm_name in ['actor_lstm', 'critic_lstm']:
            if hasattr(neural_arch, lstm_name):
                lstm = getattr(neural_arch, lstm_name)
                
                for name, param in lstm.named_parameters():
                    with torch.no_grad():
                        zero_count = (param.data.abs() < 1e-8).sum().item()
                        total = param.data.numel()
                        zero_ratio = zero_count / total
                        
                        if zero_ratio > 0.5:  # >50% é crítico
                            critical_components.append({
                                'component': f"{lstm_name}.{name}",
                                'zero_ratio': zero_ratio,
                                'total_params': total,
                                'severity': 'CRÍTICO' if zero_ratio > 0.8 else 'ALTO'
                            })
    
    # Relatório final
    if critical_components:
        print("\n🚨 COMPONENTES CRÍTICOS ENCONTRADOS:")
        for comp in critical_components:
            severity_emoji = "🚨" if comp['severity'] == 'CRÍTICO' else "⚠️"
            print(f"   {severity_emoji} {comp['component']}: {comp['zero_ratio']*100:.1f}% zeros ({comp['total_params']:,} params)")
        
        print(f"\n🔧 SOLUÇÃO: Usar fix_v8_heritage_lstm_initialization()")
        print("   modelo_fix = fix_v8_heritage_lstm_initialization()")
        print("   modelo_fix(seu_modelo)")
        
    else:
        print("✅ Nenhum componente crítico encontrado")

if __name__ == "__main__":
    print("🔧 Fix V8Heritage Zeros - Sistema de correção emergencial")
    print("Use: from fix_v8_heritage_zeros import fix_v8_heritage_lstm_initialization, diagnose_v8_heritage_zeros")