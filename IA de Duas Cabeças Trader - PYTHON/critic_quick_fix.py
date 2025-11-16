
# 🚀 PATCH RÁPIDO - CRÍTICO V7 FIX
# Adicionar no daytrader.py antes do model.learn()

def apply_critic_emergency_fix(model):
    '''Fix emergencial para crítico V7'''
    
    # 1. LEARNING RATE BOOST para crítico
    for name, param in model.policy.named_parameters():
        if 'critic' in name.lower() or 'value' in name.lower():
            param.requires_grad_(True)
            # Multiplier interno para gradientes do crítico
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad.data *= 2.0  # Boost gradientes
    
    # 2. VALUE LOSS WEIGHT aumentado
    if hasattr(model.policy, 'vf_coef'):
        model.policy.vf_coef = 1.0  # Aumentar de 0.5 para 1.0
    
    # 3. ENTROPY REGULARIZATION reduzida para dar espaço ao crítico
    if hasattr(model.policy, 'ent_coef'):
        model.policy.ent_coef *= 0.5  # Reduzir entropia
    
    print("🔧 Critic Emergency Fix aplicado!")

# USAR ASSIM:
# apply_critic_emergency_fix(model)
# model.learn(total_timesteps=50000)  # Treinar um pouco
        