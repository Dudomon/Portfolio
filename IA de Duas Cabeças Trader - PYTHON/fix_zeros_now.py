#!/usr/bin/env python3
"""
🚨 FIX IMEDIATO - ELIMINAR 62.3% ZEROS AGORA
Script para aplicar V11 initialization imediatamente durante treinamento
"""

import torch
import torch.nn as nn
import os
import sys
import time
import pickle

def find_and_fix_sac_model():
    """
    Encontrar o modelo SAC em execução e aplicar fix V11
    """
    print("🔍 BUSCANDO MODELO SAC EM EXECUÇÃO...")
    
    # Tentar encontrar arquivo de modelo temporário/checkpoint
    possible_paths = [
        "Otimizacao/treino_principal/models/SACVERSION",
        ".",
        "models",
        "checkpoints"
    ]
    
    model_found = False
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"📂 Verificando: {path}")
            
            # Procurar arquivos .zip ou .pkl
            for file in os.listdir(path):
                if file.endswith('.zip') or file.endswith('.pkl'):
                    full_path = os.path.join(path, file)
                    print(f"🔍 Encontrado: {full_path}")
                    
                    try:
                        # Tentar carregar modelo
                        if file.endswith('.zip'):
                            from stable_baselines3 import SAC
                            model = SAC.load(full_path)
                        else:
                            with open(full_path, 'rb') as f:
                                model = pickle.load(f)
                        
                        print(f"✅ Modelo carregado: {type(model)}")
                        
                        # Aplicar fix V11
                        if apply_v11_fix_to_model(model, full_path):
                            model_found = True
                            print(f"🎯 FIX APLICADO EM: {full_path}")
                            break
                            
                    except Exception as e:
                        print(f"❌ Erro ao carregar {full_path}: {e}")
                        continue
            
            if model_found:
                break
    
    if not model_found:
        print("❌ NENHUM MODELO SAC ENCONTRADO PARA APLICAR FIX")
        print("💡 ALTERNATIVA: Aplicar fix via memory manipulation se possível")
        
    return model_found

def apply_v11_fix_to_model(model, model_path):
    """
    Aplicar V11 initialization fix ao modelo
    """
    try:
        if not hasattr(model, 'policy') or not hasattr(model.policy, 'actor'):
            print("❌ Modelo não tem policy.actor")
            return False
        
        # Encontrar primeira camada
        first_layer = None
        first_layer_name = None
        
        if hasattr(model.policy.actor, 'latent_pi'):
            try:
                first_layer = model.policy.actor.latent_pi[0]
                first_layer_name = 'actor.latent_pi.0'
            except:
                pass
        
        if first_layer is None:
            for name, layer in model.policy.actor.named_modules():
                if isinstance(layer, nn.Linear):
                    first_layer = layer
                    first_layer_name = name
                    break
        
        if first_layer is None or not isinstance(first_layer, nn.Linear):
            print("❌ Primeira camada Linear não encontrada")
            return False
        
        # Verificar zeros
        zeros_before = (first_layer.weight.data == 0).float().mean().item() * 100
        print(f"🔍 {first_layer_name}: {zeros_before:.1f}% zeros ANTES")
        
        if zeros_before > 50:
            # Aplicar V11 initialization
            with torch.no_grad():
                torch.nn.init.xavier_uniform_(first_layer.weight, gain=1.0)
                if first_layer.bias is not None:
                    torch.nn.init.zeros_(first_layer.bias)
            
            zeros_after = (first_layer.weight.data == 0).float().mean().item() * 100
            print(f"✅ {first_layer_name}: {zeros_before:.1f}% → {zeros_after:.1f}% zeros")
            
            # Salvar modelo corrigido
            try:
                backup_path = model_path + ".backup"
                os.rename(model_path, backup_path)
                print(f"💾 Backup criado: {backup_path}")
                
                if model_path.endswith('.zip'):
                    model.save(model_path)
                else:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
                print(f"💾 Modelo corrigido salvo: {model_path}")
                return True
                
            except Exception as e:
                print(f"❌ Erro ao salvar modelo corrigido: {e}")
                return False
        else:
            print(f"✅ {first_layer_name}: {zeros_before:.1f}% zeros - OK")
            return True
            
    except Exception as e:
        print(f"❌ Erro ao aplicar fix: {e}")
        return False

def create_runtime_fix_signal():
    """
    Criar sinal para aplicar fix durante treinamento
    """
    signal_file = "apply_v11_fix.signal"
    
    with open(signal_file, "w") as f:
        f.write(f"V11_FIX_REQUEST_{int(time.time())}\n")
        f.write("ZERO_PERCENTAGE: 62.3%\n")
        f.write("TARGET_LAYER: actor.latent_pi.0.weight\n")
        f.write("FIX_TYPE: xavier_uniform_\n")
    
    print(f"🚨 SINAL CRIADO: {signal_file}")
    print("💡 Se o callback estiver funcionando, ele detectará este arquivo")
    
    return signal_file

if __name__ == "__main__":
    print("🚨 FIX IMEDIATO PARA ELIMINAR 62.3% ZEROS")
    print("=" * 60)
    
    # Tentar fix direto em modelo
    if not find_and_fix_sac_model():
        print("\n🚨 PLANO B: Criar sinal para callback")
        signal_file = create_runtime_fix_signal()
        
        print(f"\n💡 INSTRUÇÕES:")
        print(f"1. Arquivo de sinal criado: {signal_file}")
        print(f"2. O callback V11 deve detectar e aplicar fix")
        print(f"3. Se não funcionar, verificar se callback está ativo")
        print(f"4. Alternativa: Reiniciar treinamento com fix aplicado")
    
    print("\n✅ SCRIPT CONCLUÍDO")