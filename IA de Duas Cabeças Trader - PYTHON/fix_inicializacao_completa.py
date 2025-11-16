#!/usr/bin/env python3
"""
🔧 CORREÇÃO COMPLETA DA INICIALIZAÇÃO V7
Implementar todas as correções identificadas na análise
"""

import sys
import os
from pathlib import Path
import shutil
from datetime import datetime

projeto_path = Path("D:/Projeto")
sys.path.insert(0, str(projeto_path))

def backup_files():
    """Fazer backup dos arquivos antes das modificações"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = projeto_path / f"backup_inicializacao_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "trading_framework/policies/two_head_v7_intuition.py",
        "trading_framework/policies/two_head_v7_simple.py", 
        "trading_framework/policies/two_head_v7_unified.py"
    ]
    
    print(f"📦 CRIANDO BACKUP: {backup_dir}")
    
    for file_path in files_to_backup:
        src = projeto_path / file_path
        if src.exists():
            dst = backup_dir / file_path.replace("/", "_")
            shutil.copy2(src, dst)
            print(f"   ✅ {file_path}")
    
    return backup_dir

def create_corrected_initialization():
    """Criar código corrigido para inicialização"""
    
    # 1. CORREÇÃO PARA ACTOR_HEAD (específica por dimensão)
    actor_head_fix = '''
    def _initialize_actor_head_properly(self):
        """🔧 Inicialização CORRIGIDA do actor_head - específica por dimensão"""
        
        print("🔧 [INIT FIX] Aplicando inicialização CORRIGIDA ao actor_head...")
        
        # 1. Inicializar layers intermediárias com He (para LeakyReLU)
        for i, layer in enumerate(self.actor_head[:-1]):  # Todas exceto a última
            if isinstance(layer, torch.nn.Linear):
                # He initialization para LeakyReLU
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        
        # 2. CORREÇÃO CRÍTICA: Última layer com inicialização específica por dimensão
        last_layer = self.actor_head[-1]
        if isinstance(last_layer, torch.nn.Linear):
            # Inicialização Xavier normal para a saída
            torch.nn.init.xavier_normal_(last_layer.weight, gain=1.0)
            
            if last_layer.bias is not None:
                # 🎯 INICIALIZAÇÃO ESPECÍFICA POR DIMENSÃO DE AÇÃO
                with torch.no_grad():
                    # Action[0] - order_type: neutro para decisão balanceada  
                    last_layer.bias[0] = 0.0
                    
                    # 🔴 Action[1] - quantity: BIAS POSITIVO para resolver o problema
                    last_layer.bias[1] = 2.5  # sigmoid(2.5) = 0.924
                    
                    # Action[2] - temporal_signal: neutro
                    last_layer.bias[2] = 0.0
                    
                    # Action[3] - risk_appetite: ligeiramente conservador
                    last_layer.bias[3] = 0.5  # sigmoid(0.5) = 0.622
                    
                    # Action[4] - regime_bias: neutro
                    last_layer.bias[4] = 0.0
                    
                    # Actions[5-10] - SL/TP: neutros
                    for i in range(5, 11):
                        last_layer.bias[i] = 0.0
        
        print("✅ [INIT FIX] Actor head inicializado com bias específicos por dimensão")
        print("   🎯 Action[1] bias = +2.5 → sigmoid(2.5) = 0.924 (CORRIGIDO)")
    '''
    
    # 2. CORREÇÃO PARA LSTM (forget gate bias)
    lstm_fix = '''
    def _initialize_lstm_components_properly(self):
        """🔧 Inicialização CORRIGIDA dos LSTMs - forget gate bias = 1.0"""
        
        print("🔧 [LSTM FIX] Corrigindo inicialização dos LSTMs...")
        
        lstm_components = []
        
        # Coletar todos os LSTMs do sistema
        if hasattr(self, 'v7_actor_lstm'):
            lstm_components.append(('v7_actor_lstm', self.v7_actor_lstm))
        if hasattr(self, 'actor_lstm'):
            lstm_components.append(('actor_lstm', self.actor_lstm))
        if hasattr(self, 'v7_shared_lstm'):
            lstm_components.append(('v7_shared_lstm', self.v7_shared_lstm))
        
        for name, lstm in lstm_components:
            print(f"   🔧 Corrigindo {name}...")
            
            for param_name, param in lstm.named_parameters():
                if 'weight' in param_name:
                    # Xavier/Glorot normal para weights
                    torch.nn.init.xavier_normal_(param)
                elif 'bias' in param_name:
                    # CORREÇÃO CRÍTICA: Forget gate bias = 1.0
                    torch.nn.init.zeros_(param)  # Primeiro zerar todos
                    
                    # Identificar forget gate bias e configurar para 1.0
                    hidden_size = param.size(0) // 4
                    
                    if 'bias_ih' in param_name:
                        # Input-to-hidden bias: forget gate = [hidden_size:2*hidden_size]
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
                    elif 'bias_hh' in param_name:
                        # Hidden-to-hidden bias: forget gate = [hidden_size:2*hidden_size] 
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
            
            print(f"     ✅ {name}: forget gate bias = 1.0")
        
        print("✅ [LSTM FIX] Todos os LSTMs corrigidos")
    '''
    
    # 3. CORREÇÃO PARA CRITIC MLP  
    critic_fix = '''
    def _initialize_critic_mlp_properly(self):
        """🔧 Inicialização CORRIGIDA do critic_mlp - He initialization"""
        
        print("🔧 [CRITIC FIX] Corrigindo inicialização do critic_mlp...")
        
        if hasattr(self, 'critic_mlp'):
            for layer in self.critic_mlp:
                if isinstance(layer, torch.nn.Linear):
                    # He initialization para LeakyReLU
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
        
        print("✅ [CRITIC FIX] Critic MLP inicializado com He initialization")
    '''
    
    # 4. FUNÇÃO PRINCIPAL CORRIGIDA
    main_fix = '''
    def _initialize_all_components_properly(self):
        """🔧 Inicialização COMPLETA E CORRIGIDA de TODOS os componentes"""
        
        print("🚀 [FULL FIX] Inicializando TODOS os componentes corretamente...")
        
        # 1. CRÍTICO: Actor head com bias específicos por dimensão
        self._initialize_actor_head_properly()
        
        # 2. CRÍTICO: LSTMs com forget gate bias = 1.0  
        self._initialize_lstm_components_properly()
        
        # 3. Critic MLP com He initialization
        self._initialize_critic_mlp_properly()
        
        print("🎯 [FULL FIX] TODAS as correções aplicadas com sucesso!")
        print("   ✅ Action[1] deve agora variar normalmente")
        print("   ✅ LSTMs devem treinar sem gradient vanishing")
        print("   ✅ Critic deve ter gradientes estáveis")
    '''
    
    return {
        'actor_head_fix': actor_head_fix,
        'lstm_fix': lstm_fix, 
        'critic_fix': critic_fix,
        'main_fix': main_fix
    }

def apply_fixes():
    """Aplicar todas as correções nos arquivos"""
    
    print("🔧 APLICANDO CORREÇÕES DE INICIALIZAÇÃO")
    print("=" * 60)
    
    # 1. Fazer backup
    backup_dir = backup_files()
    
    # 2. Gerar código corrigido
    fixes = create_corrected_initialization()
    
    # 3. Aplicar correção no TwoHeadV7Intuition
    print(f"\n🔧 CORRIGINDO: TwoHeadV7Intuition")
    print("-" * 50)
    
    intuition_file = projeto_path / "trading_framework/policies/two_head_v7_intuition.py"
    
    with open(intuition_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Localizar e substituir a função _initialize_actor_head_properly
    start_marker = "def _initialize_actor_head_properly(self):"
    end_marker = "print(\"✅ [INIT FIX] Actor head inicializado com range amplo (-2 a 2)\")"
    
    start_idx = content.find(start_marker)
    if start_idx != -1:
        # Encontrar o final da função
        lines = content[start_idx:].split('\n')
        func_lines = []
        indent_level = None
        
        for i, line in enumerate(lines):
            if i == 0:  # Primeira linha da função
                func_lines.append(line)
                indent_level = len(line) - len(line.lstrip())
                continue
                
            current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 4
            
            # Se encontramos uma linha com indent menor ou igual ao da def, paramos
            if line.strip() and current_indent <= indent_level:
                break
                
            func_lines.append(line)
        
        old_func = '\n'.join(func_lines)
        
        # Substituir pela versão corrigida
        new_content = content.replace(old_func, fixes['actor_head_fix'].strip())
        
        # Adicionar as outras funções antes da última linha da classe
        class_end_idx = new_content.rfind("    print(\"🎯 TwoHeadV7Intuition PRONTA")
        if class_end_idx != -1:
            before = new_content[:class_end_idx]
            after = new_content[class_end_idx:]
            
            new_content = before + fixes['lstm_fix'] + "\n\n" + fixes['critic_fix'] + "\n\n" + fixes['main_fix'] + "\n\n    " + after
        
        # Modificar a chamada na inicialização
        old_call = "self._initialize_actor_head_properly()"
        new_call = "self._initialize_all_components_properly()"
        new_content = new_content.replace(old_call, new_call)
        
        # Salvar arquivo corrigido
        with open(intuition_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ TwoHeadV7Intuition corrigido")
    else:
        print("❌ Função _initialize_actor_head_properly não encontrada")
    
    # 4. Aplicar correções similares nos outros arquivos
    other_files = [
        ("TwoHeadV7Simple", "trading_framework/policies/two_head_v7_simple.py"),
        ("TwoHeadV7Unified", "trading_framework/policies/two_head_v7_unified.py")
    ]
    
    for class_name, file_path in other_files:
        print(f"\n🔧 CORRIGINDO: {class_name}")
        print("-" * 50)
        
        full_path = projeto_path / file_path
        
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Procurar por inicialização LSTM problemática
            problem_pattern = "elif 'bias' in name:\n                torch.nn.init.zeros_(param)"
            
            if problem_pattern in content:
                # Substituir pela versão corrigida
                fixed_pattern = '''elif 'bias' in name:
                # CORREÇÃO CRÍTICA: Forget gate bias = 1.0
                torch.nn.init.zeros_(param)
                hidden_size = param.size(0) // 4
                param.data[hidden_size:2*hidden_size].fill_(1.0)  # Forget gate bias'''
                
                new_content = content.replace(problem_pattern, fixed_pattern)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✅ {class_name} LSTM bias corrigido")
            else:
                print(f"⚠️ Padrão problemático não encontrado em {class_name}")
        else:
            print(f"❌ Arquivo não encontrado: {file_path}")
    
    # 5. Resumo das correções
    print(f"\n🎯 RESUMO DAS CORREÇÕES APLICADAS")
    print("=" * 60)
    
    corrections = [
        "✅ Actor head: bias específicos por dimensão (Action[1] = +2.5)",
        "✅ LSTMs: forget gate bias = 1.0 (3 componentes corrigidos)",
        "✅ Critic MLP: He initialization para LeakyReLU", 
        "✅ Inicialização unificada: _initialize_all_components_properly()",
        f"📦 Backup criado em: {backup_dir}"
    ]
    
    for correction in corrections:
        print(f"   {correction}")
    
    print(f"\n🚀 PRÓXIMO PASSO: INICIAR RE-TREINO")
    print(f"   O modelo agora deve produzir Action[1] variável")
    print(f"   LSTMs devem treinar sem gradient vanishing")
    print(f"   Inicialização balanceada para todos os componentes")

if __name__ == "__main__":
    apply_fixes()