#!/usr/bin/env python3
"""
🧪 Teste do sistema de avaliação corrigido
"""

import os
import glob

def test_evaluation_system():
    """Testar se o sistema de avaliação encontra checkpoints existentes"""
    
    # Simular o código da função _run_avaliar_v8_evaluation
    DIFF_MODEL_DIR = "Otimizacao/treino_principal/models/SILUS"
    checkpoint_dir = f"D:/Projeto/{DIFF_MODEL_DIR}"
    checkpoint_pattern = f"{checkpoint_dir}/*_steps_*.zip"
    
    print(f"🔍 Procurando checkpoints em: {checkpoint_pattern}")
    
    # Encontrar checkpoints existentes
    existing_checkpoints = glob.glob(checkpoint_pattern)
    
    if not existing_checkpoints:
        print("⚠️ Nenhum checkpoint encontrado para avaliação na pasta SILUS")
        
        # Tentar procurar em outras pastas
        backup_pattern = "D:/Projeto/Otimizacao/treino_principal/models/*/*_steps_*.zip"
        backup_checkpoints = glob.glob(backup_pattern)
        
        if backup_checkpoints:
            print(f"✅ Encontrados {len(backup_checkpoints)} checkpoints em outras pastas:")
            for cp in backup_checkpoints[:5]:
                print(f"  - {os.path.basename(cp)}")
            
            # Usar o mais recente
            latest_checkpoint = max(backup_checkpoints, key=os.path.getctime)
            print(f"📊 Checkpoint mais recente: {os.path.basename(latest_checkpoint)}")
        else:
            print("❌ Nenhum checkpoint encontrado em lugar algum")
        return
    
    # Usar o checkpoint mais recente
    latest_checkpoint = max(existing_checkpoints, key=os.path.getctime)
    print(f"✅ Encontrados {len(existing_checkpoints)} checkpoints na pasta SILUS")
    print(f"📊 Usando checkpoint mais recente: {os.path.basename(latest_checkpoint)}")
    
    # Verificar se avaliar_v11.py existe
    avaliar_path = "D:/Projeto/avaliacao/avaliar_v11.py"
    if os.path.exists(avaliar_path):
        print(f"✅ avaliar_v11.py encontrado")
    else:
        print(f"❌ avaliar_v11.py não encontrado em: {avaliar_path}")

if __name__ == "__main__":
    test_evaluation_system()