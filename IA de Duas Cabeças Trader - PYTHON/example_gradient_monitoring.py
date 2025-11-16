#!/usr/bin/env python3
"""
🎯 EXEMPLO COMPLETO DE MONITORAMENTO DE GRADIENTES
Demonstra como integrar o sistema de monitoramento com TwoHeadV6
"""

import sys
import os
import torch
import numpy as np

# Adicionar paths necessários
sys.path.append(".")
sys.path.append("trading_framework/policies")

def test_gradient_monitoring():
    """🧪 Testar sistema completo de monitoramento"""
    print("🔍 TESTANDO SISTEMA DE MONITORAMENTO DE GRADIENTES")
    print("=" * 60)
    
    try:
        # Importar TwoHeadV6
        from trading_framework.policies.two_head_v6_intelligent_48h import TwoHeadV6Intelligent48h
        from gradient_callback import create_gradient_callback
        import gym
        from gym import spaces
        
        # Criar policy
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1480,), dtype=np.float32)
        action_space = spaces.Discrete(64)
        
        def lr_schedule(progress):
            return 3e-4
        
        policy = TwoHeadV6Intelligent48h(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            lstm_hidden_size=128
        )
        
        print("✅ TwoHeadV6 criada com sucesso")
        
        # Configurar monitoramento de gradientes
        success = policy.setup_gradient_monitoring(
            check_frequency=10,  # Verificar a cada 10 steps para teste
            log_dir="test_gradient_logs"
        )
        
        if success:
            print("✅ Gradient Health Monitor configurado")
        else:
            print("❌ Falha ao configurar Gradient Health Monitor")
            return False
        
        # Simular treinamento com monitoramento
        print("\n🏃 Simulando treinamento com monitoramento...")
        
        for step in range(50):
            # Simular forward pass
            batch_size = 4
            obs = torch.randn(batch_size, 1480)
            lstm_states = policy.get_initial_state(batch_size)
            episode_starts = torch.zeros(batch_size, dtype=torch.bool)
            
            # Forward pass
            actions, values, log_probs, new_states = policy.forward(
                obs, lstm_states, episode_starts
            )
            
            # Simular loss
            target_values = torch.randn_like(values)
            target_actions = torch.randint(0, 64, (batch_size,))
            
            value_loss = torch.nn.MSELoss()(values, target_values)
            action_loss = torch.nn.CrossEntropyLoss()(
                log_probs.unsqueeze(1).expand(-1, 64), 
                target_actions
            )
            total_loss = value_loss + action_loss
            
            # Backward pass
            total_loss.backward()
            
            # Verificar e corrigir gradientes
            health_report = policy.check_and_fix_gradients(step)
            
            if health_report:
                health_score = health_report.get('health_score', 1.0)
                if step % 10 == 0:
                    print(f"  Step {step:2d}: Health = {health_score:.3f}, "
                          f"Zero grads = {health_report.get('zero_gradients', 0)}, "
                          f"Problems = {len(health_report.get('problematic_layers', []))}")
            
            # Simular optimizer step
            policy.zero_grad()
        
        # Obter resumo final
        print("\n📊 RESUMO FINAL:")
        summary = policy.get_gradient_health_summary()
        print(f"  Status: {summary.get('status', 'unknown')}")
        print(f"  Saúde média: {summary.get('avg_health_score', 0):.3f}")
        print(f"  Total correções: {summary.get('total_corrections', 0)}")
        print(f"  Layers problemáticos: {len(summary.get('most_problematic_layers', []))}")
        
        # Salvar relatório
        report_file = policy.save_gradient_report()
        if report_file:
            print(f"  Relatório salvo: {report_file}")
        
        print("\n✅ TESTE CONCLUÍDO COM SUCESSO!")
        return True
        
    except Exception as e:
        print(f"❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_integration_example():
    """📋 Mostrar exemplo de integração com treinamento real"""
    print("\n" + "=" * 60)
    print("📋 EXEMPLO DE INTEGRAÇÃO COM TREINAMENTO REAL")
    print("=" * 60)
    
    example_code = '''
# 1. Importar callback
from gradient_callback import create_gradient_callback

# 2. Criar callback de gradientes
gradient_callback = create_gradient_callback(
    check_frequency=500,      # Verificar a cada 500 steps
    auto_fix=True,           # Aplicar correções automáticas
    alert_threshold=0.3,     # Alertar se saúde < 30%
    log_dir="gradient_logs", # Diretório para logs
    verbose=1                # Nível de logging
)

# 3. Integrar com treinamento
model = RecurrentPPO(
    TwoHeadV6Intelligent48h,
    env,
    **ppo_kwargs
)

# 4. Treinar com monitoramento
model.learn(
    total_timesteps=1000000,
    callback=[gradient_callback]  # Adicionar callback
)

# 5. Verificar resultados
stats = gradient_callback.get_monitoring_stats()
print(f"Correções aplicadas: {stats['total_corrections']}")
print(f"Alertas críticos: {stats['critical_alerts']}")
'''
    
    print(example_code)
    print("=" * 60)
    print("✅ INTEGRAÇÃO SIMPLES E AUTOMÁTICA!")

def main():
    """Executar todos os testes"""
    print("🚀 SISTEMA DE MONITORAMENTO DE GRADIENTES")
    print("Garantindo qualidade máxima dos gradientes durante treinamento")
    print()
    
    # Teste do sistema
    success = test_gradient_monitoring()
    
    if success:
        # Mostrar exemplo de integração
        show_integration_example()
        
        print("\n🎉 SISTEMA PRONTO PARA USO!")
        print()
        print("BENEFÍCIOS:")
        print("✅ Detecção automática de gradientes problemáticos")
        print("✅ Correção automática de NaN/Inf")
        print("✅ Gradient clipping inteligente")
        print("✅ Alertas em tempo real")
        print("✅ Relatórios detalhados")
        print("✅ Integração transparente com treinamento")
        
    else:
        print("\n❌ PROBLEMAS DETECTADOS - Verificar configuração")

if __name__ == "__main__":
    main()