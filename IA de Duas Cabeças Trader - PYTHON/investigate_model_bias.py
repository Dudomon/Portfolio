#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO: Por que o modelo produz valores sempre 0.0-0.9?

Vamos descobrir a causa raiz real ao invés de fazer gambiarra
"""

import torch
import numpy as np

def analyze_model_initialization():
    """Analisar inicialização do modelo"""
    
    print("🔍 INVESTIGANDO INICIALIZAÇÃO DO MODELO")
    print("=" * 60)
    
    print("🤔 POSSÍVEIS CAUSAS:")
    print("1. 🎯 INICIALIZAÇÃO RUIM: Pesos inicializados com bias")
    print("2. 🔧 ATIVAÇÃO ERRADA: Sigmoid/Tanh limitando range")
    print("3. 📊 NORMALIZAÇÃO: Input/output sendo normalizado")
    print("4. 🧠 ARQUITETURA: Actor head com problema estrutural")
    print("5. ⚡ GRADIENTES: Saturação impedindo aprendizado")

def check_policy_architecture():
    """Verificar arquitetura da política V7"""
    
    print("\n🔍 VERIFICANDO ARQUITETURA DA POLÍTICA V7")
    print("=" * 60)
    
    try:
        # Ler código da política
        with open('trading_framework/policies/two_head_v7_intuition.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Procurar por ativações suspeitas
        suspicious_patterns = [
            ('torch.sigmoid', 'Sigmoid pode limitar saída a 0-1'),
            ('torch.tanh', 'Tanh pode limitar saída a -1,1'),
            ('nn.Sigmoid', 'Sigmoid layer limitando range'),
            ('nn.Tanh', 'Tanh layer limitando range'),
            ('F.sigmoid', 'Functional sigmoid limitando'),
            ('F.tanh', 'Functional tanh limitando'),
        ]
        
        print("🔍 PROCURANDO ATIVAÇÕES SUSPEITAS:")
        found_issues = False
        
        for pattern, description in suspicious_patterns:
            if pattern in content:
                print(f"   ⚠️ ENCONTRADO: {pattern} - {description}")
                found_issues = True
        
        if not found_issues:
            print("   ✅ Nenhuma ativação suspeita encontrada")
        
        # Procurar por actor_head
        if 'actor_head' in content:
            print("\n🔍 ANALISANDO ACTOR HEAD:")
            
            # Extrair definição do actor_head
            lines = content.split('\n')
            in_actor_head = False
            actor_head_lines = []
            
            for line in lines:
                if 'self.actor_head = nn.Sequential(' in line:
                    in_actor_head = True
                    actor_head_lines.append(line.strip())
                elif in_actor_head:
                    actor_head_lines.append(line.strip())
                    if ')' in line and not line.strip().startswith('nn.'):
                        break
            
            if actor_head_lines:
                print("   📊 DEFINIÇÃO DO ACTOR HEAD:")
                for line in actor_head_lines:
                    print(f"      {line}")
                
                # Verificar última camada
                last_line = actor_head_lines[-2] if len(actor_head_lines) > 1 else ""
                if 'Linear' in last_line and 'self.action_space.shape[0]' in last_line:
                    print("   ✅ Última camada: Linear sem ativação (correto)")
                else:
                    print("   ⚠️ Última camada pode ter ativação limitante")
        
    except FileNotFoundError:
        print("❌ Arquivo da política não encontrado")

def check_action_processing():
    """Verificar processamento das ações"""
    
    print("\n🔍 VERIFICANDO PROCESSAMENTO DAS AÇÕES")
    print("=" * 60)
    
    try:
        with open('trading_framework/policies/two_head_v7_intuition.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Procurar por processamento das ações
        if 'actions[:, 0] = discrete_decision.float()' in content:
            print("✅ Entry decision sendo processado corretamente")
        
        # Procurar por raw_actions
        if 'raw_actions = self.actor_head(' in content:
            print("✅ Raw actions sendo geradas pelo actor_head")
            
            # Verificar se há clipping ou normalização
            if 'torch.clamp' in content:
                print("⚠️ ENCONTRADO: torch.clamp - pode estar limitando valores")
            
            if 'torch.sigmoid' in content or 'torch.tanh' in content:
                print("⚠️ ENCONTRADO: Ativação limitante nas ações")
        
        # Verificar inicialização
        if 'ortho_init' in content:
            print("✅ Inicialização ortogonal configurada")
        
        if 'log_std_init' in content:
            print("✅ Log std init configurado")
    
    except FileNotFoundError:
        print("❌ Arquivo não encontrado")

def suggest_proper_fixes():
    """Sugerir correções adequadas"""
    
    print("\n🔧 CORREÇÕES ADEQUADAS (NÃO GAMBIARRAS)")
    print("=" * 60)
    
    fixes = [
        {
            "issue": "Inicialização ruim",
            "fix": "Usar Xavier/He initialization no actor_head",
            "code": "nn.init.xavier_uniform_(self.actor_head[-1].weight)"
        },
        {
            "issue": "Ativação limitante",
            "fix": "Remover sigmoid/tanh da última camada",
            "code": "# Última camada deve ser Linear puro"
        },
        {
            "issue": "Range inadequado",
            "fix": "Ajustar inicialização para range maior",
            "code": "nn.init.uniform_(layer.weight, -2.0, 2.0)"
        },
        {
            "issue": "Saturação de gradientes",
            "fix": "Usar LeakyReLU ao invés de ReLU",
            "code": "nn.LeakyReLU(negative_slope=0.01)"
        },
        {
            "issue": "Normalização excessiva",
            "fix": "Verificar se VecNormalize está afetando ações",
            "code": "# VecNormalize deve normalizar obs, não actions"
        }
    ]
    
    print("📋 LISTA DE CORREÇÕES ADEQUADAS:")
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['issue']}:")
        print(f"   Solução: {fix['fix']}")
        print(f"   Código: {fix['code']}")

def create_proper_investigation():
    """Criar investigação adequada"""
    
    investigation_code = '''
# 🔍 INVESTIGAÇÃO ADEQUADA - Adicionar na política V7

def debug_raw_actions(self, raw_actions):
    """Debug das ações brutas antes do processamento"""
    
    print(f"🔍 [RAW ACTIONS DEBUG]:")
    print(f"   Shape: {raw_actions.shape}")
    print(f"   Min: {raw_actions.min().item():.4f}")
    print(f"   Max: {raw_actions.max().item():.4f}")
    print(f"   Mean: {raw_actions.mean().item():.4f}")
    print(f"   Std: {raw_actions.std().item():.4f}")
    
    # Verificar distribuição
    values = raw_actions.detach().cpu().numpy().flatten()
    
    ranges = [
        ("< -2", np.sum(values < -2)),
        ("-2 a -1", np.sum((values >= -2) & (values < -1))),
        ("-1 a 0", np.sum((values >= -1) & (values < 0))),
        ("0 a 1", np.sum((values >= 0) & (values < 1))),
        ("1 a 2", np.sum((values >= 1) & (values < 2))),
        ("> 2", np.sum(values >= 2)),
    ]
    
    total = len(values)
    print(f"   Distribuição:")
    for range_name, count in ranges:
        pct = (count / total) * 100
        print(f"     {range_name}: {count} ({pct:.1f}%)")
    
    return raw_actions

# Usar no forward_actor:
# raw_actions = self.actor_head(actor_input)
# raw_actions = self.debug_raw_actions(raw_actions)  # ADICIONAR ESTA LINHA
'''
    
    with open('proper_investigation_patch.py', 'w', encoding='utf-8') as f:
        f.write(investigation_code)
    
    print(f"\n💾 Investigação adequada criada: proper_investigation_patch.py")

if __name__ == "__main__":
    print("🔍 INVESTIGAÇÃO DA CAUSA RAIZ REAL")
    print("=" * 80)
    print("Você está certo - é gambiarra ajustar thresholds!")
    print("Vamos descobrir por que o modelo produz valores 0.0-0.9")
    print()
    
    # 1. Analisar inicialização
    analyze_model_initialization()
    
    # 2. Verificar arquitetura
    check_policy_architecture()
    
    # 3. Verificar processamento
    check_action_processing()
    
    # 4. Sugerir correções adequadas
    suggest_proper_fixes()
    
    # 5. Criar investigação adequada
    create_proper_investigation()
    
    print("\n🎯 CONCLUSÃO:")
    print("A gambiarra funciona, mas não resolve a causa raiz.")
    print("O modelo DEVERIA produzir valores em range maior (-3 a 3).")
    print("Algo na arquitetura/inicialização está limitando os valores.")
    
    print("\n🚀 PRÓXIMO PASSO ADEQUADO:")
    print("1. Aplicar proper_investigation_patch.py na política")
    print("2. Ver distribuição real dos raw_actions")
    print("3. Corrigir a causa raiz ao invés de ajustar thresholds")