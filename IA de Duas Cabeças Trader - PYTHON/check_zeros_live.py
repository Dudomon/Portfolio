#!/usr/bin/env python3
"""
🔍 MONITOR ZEROS EM TEMPO REAL
Conecta no treinamento SAC ativo e monitora zeros
"""

import torch
import gc
import time
import os

def find_sac_model():
    """Encontrar modelo SAC ativo na memória"""
    for obj in gc.get_objects():
        try:
            if hasattr(obj, 'policy') and hasattr(obj.policy, 'actor'):
                if hasattr(obj.policy.actor, 'latent_pi'):
                    return obj
        except:
            continue
    return None

def check_zeros_now():
    """Check zeros agora no modelo ativo"""
    model = find_sac_model()
    if not model:
        print("❌ Modelo SAC não encontrado na memória")
        return
    
    print(f"🔍 ZEROS CHECK - {time.strftime('%H:%M:%S')}")
    
    # Actor
    try:
        first_layer = model.policy.actor.latent_pi[0]
        if hasattr(first_layer, 'weight'):
            weight_data = first_layer.weight.data
            zeros_count = (weight_data == 0).sum().item()
            total_count = weight_data.numel()
            zeros_pct = (zeros_count / total_count) * 100
            print(f"   ACTOR: {zeros_pct:.1f}% zeros ({zeros_count}/{total_count})")
        else:
            print("   ACTOR: Sem weight attribute")
    except Exception as e:
        print(f"   ACTOR: Erro - {e}")
    
    # Critics
    for critic_name in ['qf0', 'qf1']:
        try:
            if hasattr(model.policy, critic_name):
                critic_net = getattr(model.policy, critic_name)
                if hasattr(critic_net, '0') and hasattr(critic_net[0], 'weight'):
                    critic_layer = critic_net[0]
                    weight_data = critic_layer.weight.data
                    zeros_count = (weight_data == 0).sum().item()
                    total_count = weight_data.numel()
                    zeros_pct = (zeros_count / total_count) * 100
                    print(f"   {critic_name.upper()}: {zeros_pct:.1f}% zeros ({zeros_count}/{total_count})")
                else:
                    print(f"   {critic_name.upper()}: Estrutura diferente")
            else:
                print(f"   {critic_name.upper()}: Não encontrado")
        except Exception as e:
            print(f"   {critic_name.upper()}: Erro - {e}")

if __name__ == "__main__":
    print("🔍 MONITOR ZEROS - Checando a cada 10 segundos")
    print("   Ctrl+C para parar")
    
    try:
        while True:
            check_zeros_now()
            print()
            time.sleep(10)
    except KeyboardInterrupt:
        print("✅ Monitor parado")