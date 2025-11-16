#!/usr/bin/env python3
"""
ANÁLISE CRÍTICA DOS HIPERPARÂMETROS VS COMPLEXIDADE ARQUITETURA V7
"""

def analyze_hyperparameters():
    print("[ANÁLISE CRÍTICA] HIPERPARÂMETROS VS COMPLEXIDADE V7")
    print("=" * 80)
    
    # Dados da arquitetura atual
    current_hyperparams = {
        'learning_rate_real': 1.0e-04,      # lr_schedule base
        'learning_rate_config': 2.0e-04,    # BEST_PARAMS (não usado)
        'n_steps': 2048,                    # Rollout buffer
        'batch_size': 32,                   # Mini-batch size
        'n_epochs': 4,                      # Epochs per update
        'ent_coef': 0.05,                   # Entropy coefficient
        'clip_range': 0.3,                  # PPO clip range
        'max_grad_norm': 10.0,              # Gradient clipping
        'vf_coef': 0.8,                     # Value function coefficient
        'gamma': 0.99,                      # Discount factor
        'gae_lambda': 0.95                  # GAE lambda
    }
    
    # Complexidade da arquitetura
    architecture = {
        'total_params': 1453568,            # ~1.45M parâmetros
        'observation_space': 2580,         # 129 features x 20 timesteps
        'action_space': 11,                 # 11 dimensões de ação
        'lstm_hidden': 128,                 # LSTM hidden size
        'lstm_layers': 2,                   # 2 camadas LSTM
        'attention_heads': 4,               # 4 cabeças de atenção
        'shared_backbone': 512,             # Backbone unificado
        'dataset_size': 2000000             # 2M barras dataset
    }
    
    # Calcular métricas críticas
    effective_batch_size = current_hyperparams['n_steps'] * current_hyperparams['n_epochs']
    params_per_batch = architecture['total_params'] / effective_batch_size
    lr_per_param = current_hyperparams['learning_rate_real'] / architecture['total_params']
    
    print("MÉTRICAS ATUAIS:")
    print(f"  LR Real: {current_hyperparams['learning_rate_real']:.2e}")
    print(f"  Batch Efetivo: {effective_batch_size:,} updates")
    print(f"  Parâmetros: {architecture['total_params']:,}")
    print(f"  LR/Parâmetro: {lr_per_param:.2e}")
    print(f"  Parâmetros/Batch: {params_per_batch:.1f}")
    
    print("\n[PROBLEMAS IDENTIFICADOS]")
    problems = []
    
    # 1. Learning Rate vs Complexidade  
    if lr_per_param < 1e-10:
        problems.append("LR muito baixo para modelo complexo (1.45M params)")
        print("❌ LR/Parâmetro: {:.2e} (muito baixo!)".format(lr_per_param))
    
    # 2. Batch Size vs Parâmetros
    if params_per_batch > 100:
        problems.append("Batch size muito pequeno para modelo grande")
        print("❌ Parâmetros/Batch: {:.1f} (ratio alta!)".format(params_per_batch))
    
    # 3. Entropy Coefficient
    if current_hyperparams['ent_coef'] < 0.1:
        problems.append("Entropy coeff baixo demais para exploração")
        print("❌ Entropy Coeff: {} (baixo para modelo complexo)".format(current_hyperparams['ent_coef']))
    
    # 4. Clip Range vs LR
    clip_lr_ratio = current_hyperparams['clip_range'] / current_hyperparams['learning_rate_real']
    if clip_lr_ratio > 5000:
        problems.append("Clip range muito alto vs LR (limita updates)")
        print("❌ Clip/LR Ratio: {:.0f} (muito alto!)".format(clip_lr_ratio))
    
    # 5. N_epochs vs Complexidade
    if current_hyperparams['n_epochs'] < 6:
        problems.append("Poucas epochs para modelo complexo aprender")
        print("❌ N_epochs: {} (baixo para 1.45M params)".format(current_hyperparams['n_epochs']))
    
    # 6. Dataset Exposure
    steps_per_epoch = architecture['dataset_size'] / current_hyperparams['n_steps']
    epochs_for_full_dataset = steps_per_epoch
    if epochs_for_full_dataset > 1000:
        problems.append("Dataset muito grande vs batch size (exposição baixa)")
        print("❌ Steps para 1 época: {:.0f} (muito alto!)".format(epochs_for_full_dataset))
    
    print(f"\n[RESUMO] {len(problems)} PROBLEMAS CRÍTICOS IDENTIFICADOS")
    
    return problems, current_hyperparams, architecture

def suggest_fixes(problems, current_hyperparams, architecture):
    print("\n[CORREÇÕES SUGERIDAS]")
    print("=" * 80)
    
    suggested = {}
    
    # 1. Learning Rate: Aumentar para modelo complexo
    suggested['learning_rate'] = 3.0e-04  # 3x maior
    print(f"1. Learning Rate: {current_hyperparams['learning_rate_real']:.2e} → {suggested['learning_rate']:.2e}")
    print("   Razão: Modelo 1.45M params precisa LR maior para convergir")
    
    # 2. Batch Size: Aumentar para estabilidade
    suggested['batch_size'] = 64  # 2x maior
    print(f"2. Batch Size: {current_hyperparams['batch_size']} → {suggested['batch_size']}")
    print("   Razão: Reduzir noise nos gradientes de modelo complexo")
    
    # 3. N_epochs: Aumentar para mais aprendizado
    suggested['n_epochs'] = 8  # 2x maior  
    print(f"3. N_epochs: {current_hyperparams['n_epochs']} → {suggested['n_epochs']}")
    print("   Razão: Modelo complexo precisa mais iterações por batch")
    
    # 4. Entropy Coeff: Aumentar para exploração
    suggested['ent_coef'] = 0.1  # 2x maior
    print(f"4. Entropy Coeff: {current_hyperparams['ent_coef']} → {suggested['ent_coef']}")
    print("   Razão: Prevenir entropy collapse em modelo complexo")
    
    # 5. Clip Range: Reduzir para permitir updates maiores
    suggested['clip_range'] = 0.15  # Metade
    print(f"5. Clip Range: {current_hyperparams['clip_range']} → {suggested['clip_range']}")
    print("   Razão: Permitir updates maiores com LR aumentado")
    
    # 6. N_steps: Reduzir para mais updates frequentes
    suggested['n_steps'] = 1024  # Metade
    print(f"6. N_steps: {current_hyperparams['n_steps']} → {suggested['n_steps']}")
    print("   Razão: Updates mais frequentes para dataset grande")
    
    # 7. Max Grad Norm: Ajustar para LR maior
    suggested['max_grad_norm'] = 5.0  # Reduzir
    print(f"7. Max Grad Norm: {current_hyperparams['max_grad_norm']} → {suggested['max_grad_norm']}")
    print("   Razão: Prevenir explosão com LR maior")
    
    print(f"\n[RESULTADO ESPERADO]")
    new_effective_batch = suggested['n_steps'] * suggested['n_epochs']
    new_lr_per_param = suggested['learning_rate'] / architecture['total_params']
    new_params_per_batch = architecture['total_params'] / new_effective_batch
    
    print(f"  Batch Efetivo: {new_effective_batch:,} (vs {current_hyperparams['n_steps'] * current_hyperparams['n_epochs']:,})")
    print(f"  LR/Parâmetro: {new_lr_per_param:.2e} (vs {current_hyperparams['learning_rate_real'] / architecture['total_params']:.2e})")
    print(f"  Parâmetros/Batch: {new_params_per_batch:.1f} (vs {architecture['total_params'] / (current_hyperparams['n_steps'] * current_hyperparams['n_epochs']):.1f})")
    
    return suggested

if __name__ == '__main__':
    problems, current, arch = analyze_hyperparameters()
    if problems:
        suggested = suggest_fixes(problems, current, arch)
        print(f"\n[CONCLUSÃO] Hiperparâmetros INADEQUADOS para arquitetura V7 complexa")
        print("Necessário ajuste CRÍTICO antes de retreino!")
    else:
        print("\n[CONCLUSÃO] Hiperparâmetros adequados")