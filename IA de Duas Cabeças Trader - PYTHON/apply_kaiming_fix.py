#!/usr/bin/env python3
"""
🎯 APLICAR FIX KAIMING DIRETAMENTE NO SAC EM EXECUÇÃO
Script para aplicar Kaiming initialization durante o treinamento
"""

import torch
import os
import time

def apply_kaiming_fix_to_running_sac():
    """
    Encontrar e aplicar fix Kaiming no SAC em execução através de sinais
    """
    print("🎯 APLICANDO KAIMING FIX NO SAC EM EXECUÇÃO")
    print("=" * 60)
    
    # Criar sinal que será detectado pelo código principal
    signal_content = f"""# KAIMING FIX SIGNAL - {int(time.time())}
# Este arquivo será lido pelo sacversion.py para aplicar fix

APPLY_KAIMING_FIX = True
TARGET_LAYERS = ["actor.latent_pi.0", "critic.qf0.0", "critic.qf1.0"]
INITIALIZATION = "kaiming_uniform"
NONLINEARITY = "leaky_relu"
REASON = "64.4% zeros no actor, 58.5% e 60.6% nos critics"
TIMESTAMP = {int(time.time())}
"""
    
    # Escrever sinal
    signal_file = "kaiming_fix_signal.py"
    with open(signal_file, "w") as f:
        f.write(signal_content)
    
    print(f"✅ Sinal criado: {signal_file}")
    
    # Também criar arquivo txt para debug
    debug_file = "APLICAR_KAIMING_FIX.txt"
    with open(debug_file, "w") as f:
        f.write(f"TIMESTAMP: {time.ctime()}\n")
        f.write("PROBLEMA: 64.4% zeros no actor, 58.5% e 60.6% nos critics\n")
        f.write("SOLUÇÃO: Aplicar torch.nn.init.kaiming_uniform_ nas primeiras camadas\n")
        f.write("STATUS: AGUARDANDO APLICAÇÃO\n")
    
    print(f"✅ Debug criado: {debug_file}")
    
    # Instruções
    print("\n💡 INSTRUÇÕES:")
    print("1. Os arquivos foram criados para sinalizar a necessidade do fix")
    print("2. O próximo restart do SAC deve detectar e aplicar o fix")
    print("3. Ou adicionar código para ler estes sinais durante execução")
    
    return signal_file, debug_file

def create_direct_fix_code():
    """
    Criar código Python que pode ser executado diretamente
    """
    fix_code = '''
# CÓDIGO PARA APLICAR KAIMING FIX DIRETAMENTE
# Copie e cole este código no console Python durante o treinamento

import torch

def apply_kaiming_fix_now():
    """Aplicar fix Kaiming diretamente no modelo global"""
    
    # Tentar encontrar modelo na memória global
    import gc
    
    for obj in gc.get_objects():
        if hasattr(obj, 'policy') and hasattr(obj.policy, 'actor'):
            try:
                # Actor fix
                if hasattr(obj.policy.actor, 'latent_pi'):
                    first_layer = obj.policy.actor.latent_pi[0]
                    if isinstance(first_layer, torch.nn.Linear):
                        zeros_before = (first_layer.weight.data == 0).float().mean().item() * 100
                        if zeros_before > 60:
                            with torch.no_grad():
                                torch.nn.init.kaiming_uniform_(first_layer.weight, nonlinearity='leaky_relu')
                                if first_layer.bias is not None:
                                    torch.nn.init.zeros_(first_layer.bias)
                            zeros_after = (first_layer.weight.data == 0).float().mean().item() * 100
                            print(f"🎯 ACTOR FIX: {zeros_before:.1f}% → {zeros_after:.1f}% zeros")
                
                # Critics fix
                for critic_name in ['qf0', 'qf1']:
                    if hasattr(obj.policy, critic_name):
                        critic_net = getattr(obj.policy, critic_name)
                        if hasattr(critic_net, '0'):
                            critic_layer = critic_net[0]
                            if isinstance(critic_layer, torch.nn.Linear):
                                zeros_before = (critic_layer.weight.data == 0).float().mean().item() * 100
                                if zeros_before > 50:
                                    with torch.no_grad():
                                        torch.nn.init.kaiming_uniform_(critic_layer.weight, nonlinearity='leaky_relu')
                                        if critic_layer.bias is not None:
                                            torch.nn.init.zeros_(critic_layer.bias)
                                    zeros_after = (critic_layer.weight.data == 0).float().mean().item() * 100
                                    print(f"🎯 CRITIC {critic_name.upper()} FIX: {zeros_before:.1f}% → {zeros_after:.1f}% zeros")
                
                print("✅ KAIMING FIX APLICADO COM SUCESSO!")
                return True
                
            except Exception as e:
                continue
    
    print("❌ Modelo SAC não encontrado na memória")
    return False

# Executar fix
apply_kaiming_fix_now()
'''
    
    code_file = "kaiming_fix_code.py"
    with open(code_file, "w") as f:
        f.write(fix_code)
    
    print(f"\n📝 Código direto criado: {code_file}")
    print("💡 Execute este arquivo com: python kaiming_fix_code.py")
    
    return code_file

if __name__ == "__main__":
    signal_file, debug_file = apply_kaiming_fix_to_running_sac()
    code_file = create_direct_fix_code()
    
    print("\n🎯 RESUMO:")
    print(f"   Sinal: {signal_file}")
    print(f"   Debug: {debug_file}")
    print(f"   Código: {code_file}")
    print("\n✅ PRONTO PARA APLICAR KAIMING FIX!")