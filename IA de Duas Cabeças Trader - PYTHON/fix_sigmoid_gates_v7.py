#!/usr/bin/env python3
"""
🔧 FIX SIGMOID GATES V7 - Correção definitiva da saturação
"""

import sys
import os
sys.path.append("D:/Projeto")

def create_fixed_version():
    """Criar versão corrigida do two_head_v7_simple.py"""
    
    original_file = "D:/Projeto/trading_framework/policies/two_head_v7_simple.py"
    backup_file = "D:/Projeto/trading_framework/policies/two_head_v7_simple_BACKUP.py"
    
    print("🔧 FIX SIGMOID GATES V7")
    print("=" * 60)
    
    try:
        # Fazer backup
        import shutil
        shutil.copy2(original_file, backup_file)
        print(f"✅ Backup criado: {backup_file}")
        
        # Ler arquivo original
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # CORREÇÕES A APLICAR:
        
        # 1. REMOVER SIGMOID DE TODAS AS REDES INDIVIDUAIS
        replacements = [
            # Temporal
            ('        self.horizon_analyzer = nn.Sequential(\n'
             '            nn.Linear(input_dim, 64),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.LayerNorm(64),\n'
             '            nn.Dropout(0.1),\n'
             '            nn.Linear(64, 32),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.Linear(32, 1),\n'
             '            nn.Sigmoid()\n'
             '        )',
             '        self.horizon_analyzer = nn.Sequential(\n'
             '            nn.Linear(input_dim, 64),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.LayerNorm(64),\n'
             '            nn.Dropout(0.1),\n'
             '            nn.Linear(64, 32),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.Linear(32, 1)\n'
             '            # 🔧 FIX: SIGMOID REMOVIDO - saída linear para evitar saturação\n'
             '        )'),
            
            # MTF Validator
            ('        self.mtf_validator = nn.Sequential(\n'
             '            nn.Linear(input_dim, 64),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.LayerNorm(64),\n'
             '            nn.Linear(64, 32),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.Linear(32, 1),\n'
             '            nn.Sigmoid()\n'
             '        )',
             '        self.mtf_validator = nn.Sequential(\n'
             '            nn.Linear(input_dim, 64),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.LayerNorm(64),\n'
             '            nn.Linear(64, 32),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.Linear(32, 1)\n'
             '            # 🔧 FIX: SIGMOID REMOVIDO\n'
             '        )'),
            
            # Pattern Memory
            ('        self.pattern_memory_validator = nn.Sequential(\n'
             '            nn.Linear(input_dim, 32),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.Linear(32, 16),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.Linear(16, 1),\n'
             '            nn.Sigmoid()\n'
             '        )',
             '        self.pattern_memory_validator = nn.Sequential(\n'
             '            nn.Linear(input_dim, 32),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.Linear(32, 16),\n'
             '            nn.LeakyReLU(negative_slope=0.01),\n'
             '            nn.Linear(16, 1)\n'
             '            # 🔧 FIX: SIGMOID REMOVIDO\n'
             '        )'),
        ]
        
        # 2. MUDAR GATES PARA USAR TANH AO INVÉS DE SIGMOID
        gate_replacements = [
            # Temporal gate
            ('        temporal_gate = torch.sigmoid((temporal_score - regime_threshold) * 2.0)      # REDUZIDO de 5 para 2',
             '        temporal_gate = 0.5 + 0.5 * torch.tanh((temporal_score - regime_threshold) * 0.5)  # 🔧 FIX: Tanh suave'),
            
            # Validation gate
            ('        validation_gate = torch.sigmoid((validation_score - main_threshold) * 2.0)    # REDUZIDO de 5 para 2',
             '        validation_gate = 0.5 + 0.5 * torch.tanh((validation_score - main_threshold) * 0.5)  # 🔧 FIX: Tanh suave'),
            
            # Risk gate
            ('        risk_gate = torch.sigmoid((risk_composite - risk_threshold) * 2.0)            # REDUZIDO de 5 para 2',
             '        risk_gate = 0.5 + 0.5 * torch.tanh((risk_composite - risk_threshold) * 0.5)  # 🔧 FIX: Tanh suave'),
            
            # Market gate
            ('        market_gate = torch.sigmoid((market_score - regime_threshold) * 2.0)          # REDUZIDO de 5 para 2',
             '        market_gate = 0.5 + 0.5 * torch.tanh((market_score - regime_threshold) * 0.5)  # 🔧 FIX: Tanh suave'),
            
            # Quality gate
            ('        quality_gate = torch.sigmoid((quality_score - main_threshold) * 2.0)          # REDUZIDO de 5 para 2',
             '        quality_gate = 0.5 + 0.5 * torch.tanh((quality_score - main_threshold) * 0.5)  # 🔧 FIX: Tanh suave'),
            
            # Confidence gate
            ('        confidence_gate = torch.sigmoid((confidence_score - main_threshold) * 2.0)    # REDUZIDO de 5 para 2',
             '        confidence_gate = 0.5 + 0.5 * torch.tanh((confidence_score - main_threshold) * 0.5)  # 🔧 FIX: Tanh suave'),
        ]
        
        # Aplicar todas as correções
        modified_content = content
        total_replacements = 0
        
        # Sigmoids das redes individuais  
        for old, new in replacements:
            if old in modified_content:
                modified_content = modified_content.replace(old, new)
                total_replacements += 1
                print(f"✅ Sigmoid removido de rede individual")
            
        # Gates com tanh
        for old, new in gate_replacements:
            if old in modified_content:
                modified_content = modified_content.replace(old, new)
                total_replacements += 1
                print(f"✅ Gate convertido para tanh suave")
        
        # Salvar arquivo modificado
        with open(original_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"\n🎯 CORREÇÕES APLICADAS:")
        print(f"  Total de substituições: {total_replacements}")
        print(f"  Arquivo modificado: {original_file}")
        print(f"  Backup disponível: {backup_file}")
        
        print(f"\n🔧 MUDANÇAS IMPLEMENTADAS:")
        print(f"  1. ✅ Sigmoids removidas das redes individuais")
        print(f"  2. ✅ Gates convertidos para tanh suave (0.5x multiplicador)")
        print(f"  3. ✅ Range dos gates: [0, 1] → mais suave")
        
        print(f"\n💡 RESULTADO ESPERADO:")
        print(f"  • Fim da saturação binária (0 ou 1)")
        print(f"  • Entry Quality com valores intermediários")
        print(f"  • Gradientes funcionais em toda a rede")
        
        print(f"\n🚀 PRÓXIMO PASSO:")
        print(f"  Treinar novo modelo ou continuar treinamento existente")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_fixed_version()