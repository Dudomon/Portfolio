
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
