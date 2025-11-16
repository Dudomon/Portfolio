#!/usr/bin/env python3
"""
🚀 LSTM GRADIENT OPTIMIZER
Soluções específicas para melhorar gradientes das LSTMs
"""

import torch
import torch.nn as nn
import numpy as np

class LSTMGradientOptimizer:
    """🚀 Otimizador específico para gradientes LSTM"""
    
    def __init__(self):
        self.solutions = []
    
    def analyze_lstm_problems(self):
        """🔍 Analisar problemas específicos das LSTMs"""
        print("🔍 DIAGNÓSTICO DOS PROBLEMAS LSTM")
        print("=" * 60)
        
        problems = [
            {
                'problem': 'Vanishing Gradients',
                'symptoms': ['15.43% zeros nos bias', 'Gradientes muito pequenos'],
                'causes': ['Sequências muito longas', 'Inicialização ruim', 'Learning rate baixo'],
                'severity': 'Alto'
            },
            {
                'problem': 'Exploding Gradients', 
                'symptoms': ['Gradientes instáveis', 'Loss oscilando'],
                'causes': ['Gradient clipping insuficiente', 'Learning rate alto'],
                'severity': 'Médio'
            },
            {
                'problem': 'Dead Neurons',
                'symptoms': ['Bias zeros', 'Ativações uniformes'],
                'causes': ['Saturação das ativações', 'Inicialização inadequada'],
                'severity': 'Alto'
            }
        ]
        
        for problem in problems:
            print(f"\n🚨 {problem['problem']} ({problem['severity']} severidade):")
            print(f"   Sintomas: {', '.join(problem['symptoms'])}")
            print(f"   Causas: {', '.join(problem['causes'])}")
        
        return problems
    
    def generate_solutions(self):
        """💡 Gerar soluções específicas para LSTM"""
        print(f"\n💡 SOLUÇÕES PARA MELHORAR GRADIENTES LSTM")
        print("=" * 60)
        
        solutions = [
            {
                'name': '1. GRADIENT CLIPPING INTELIGENTE',
                'description': 'Clipping adaptativo baseado na norma dos gradientes',
                'implementation': 'Ajustar max_grad_norm dinamicamente',
                'impact': 'Alto',
                'difficulty': 'Baixo',
                'code': '''
# No daytrader.py, ajustar gradient clipping:
model.learn(
    total_timesteps=steps,
    callback=callback,
    max_grad_norm=0.5,  # Reduzir de 1.0 para 0.5
    # Ou usar clipping adaptativo
)
                '''
            },
            {
                'name': '2. INICIALIZAÇÃO LSTM MELHORADA',
                'description': 'Reinicializar LSTM com Xavier/Orthogonal',
                'implementation': 'Aplicar inicialização específica para LSTM',
                'impact': 'Alto',
                'difficulty': 'Médio',
                'code': '''
def init_lstm_weights(lstm_layer):
    for name, param in lstm_layer.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            # Forget gate bias = 1 (padrão LSTM)
            n = param.size(0)
            param.data[n//4:n//2].fill_(1.0)
                '''
            },
            {
                'name': '3. LEARNING RATE ESPECÍFICO PARA LSTM',
                'description': 'LR diferenciado para layers LSTM',
                'implementation': 'Usar optimizer groups com LRs diferentes',
                'impact': 'Médio',
                'difficulty': 'Médio',
                'code': '''
# Separar parâmetros LSTM dos outros
lstm_params = []
other_params = []

for name, param in model.policy.named_parameters():
    if 'lstm' in name.lower():
        lstm_params.append(param)
    else:
        other_params.append(param)

# Optimizer com LRs diferentes
optimizer = torch.optim.Adam([
    {'params': lstm_params, 'lr': 1e-4},    # LR menor para LSTM
    {'params': other_params, 'lr': 3e-4}    # LR normal para outros
])
                '''
            },
            {
                'name': '4. LSTM COM LAYER NORMALIZATION',
                'description': 'Adicionar LayerNorm nas LSTMs',
                'implementation': 'Modificar arquitetura para incluir normalization',
                'impact': 'Alto',
                'difficulty': 'Alto',
                'code': '''
class NormalizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        output = self.layer_norm(output)
        return output, hidden
                '''
            },
            {
                'name': '5. RESIDUAL CONNECTIONS NAS LSTM',
                'description': 'Skip connections para melhor fluxo de gradientes',
                'implementation': 'Adicionar conexões residuais',
                'impact': 'Alto',
                'difficulty': 'Alto',
                'code': '''
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.projection = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
    
    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        
        # Residual connection
        if self.projection:
            residual = self.projection(x)
        else:
            residual = x
        
        return output + residual, hidden
                '''
            },
            {
                'name': '6. GRADIENT ACCUMULATION',
                'description': 'Acumular gradientes para estabilidade',
                'implementation': 'Modificar loop de treinamento',
                'impact': 'Médio',
                'difficulty': 'Baixo',
                'code': '''
# No PPO, usar gradient accumulation
accumulation_steps = 4
for step in range(0, total_steps, accumulation_steps):
    # Acumular gradientes sem aplicar
    loss = compute_loss()
    loss = loss / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        # Aplicar gradientes acumulados
        optimizer.step()
        optimizer.zero_grad()
                '''
            },
            {
                'name': '7. WARMUP DO LEARNING RATE',
                'description': 'Começar com LR baixo e aumentar gradualmente',
                'implementation': 'Scheduler de warmup',
                'impact': 'Médio',
                'difficulty': 'Baixo',
                'code': '''
def lr_schedule_with_warmup(progress):
    warmup_steps = 0.1  # 10% dos steps para warmup
    if progress < warmup_steps:
        # Warmup: aumentar LR gradualmente
        return 3e-4 * (progress / warmup_steps)
    else:
        # Decay normal após warmup
        return 3e-4 * (1 - progress)
                '''
            },
            {
                'name': '8. LSTM BIDIRECTIONAL',
                'description': 'Usar LSTM bidirecional para melhor contexto',
                'implementation': 'Modificar para bidirectional=True',
                'impact': 'Alto',
                'difficulty': 'Médio',
                'code': '''
# Substituir LSTM unidirecional por bidirecional
self.lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    bidirectional=True,  # Adicionar esta linha
    batch_first=True
)

# Ajustar dimensões de saída (hidden_size * 2)
                '''
            }
        ]
        
        for solution in solutions:
            print(f"\n{solution['name']}")
            print(f"   📝 {solution['description']}")
            print(f"   🎯 Impacto: {solution['impact']}")
            print(f"   ⚙️ Dificuldade: {solution['difficulty']}")
            print(f"   💻 Implementação: {solution['implementation']}")
        
        return solutions
    
    def recommend_immediate_actions(self):
        """🚀 Ações imediatas para implementar"""
        print(f"\n🚀 AÇÕES IMEDIATAS RECOMENDADAS")
        print("=" * 60)
        
        immediate_actions = [
            {
                'priority': 1,
                'action': 'Reduzir Gradient Clipping',
                'change': 'max_grad_norm: 1.0 → 0.5',
                'reason': 'Gradientes LSTM são mais sensíveis',
                'effort': 'Mínimo (1 linha)'
            },
            {
                'priority': 2,
                'action': 'Implementar LR Warmup',
                'change': 'Adicionar warmup de 10% dos steps',
                'reason': 'LSTM precisa de inicialização suave',
                'effort': 'Baixo (função lr_schedule)'
            },
            {
                'priority': 3,
                'action': 'Reinicializar LSTM Bias',
                'change': 'Forget gate bias = 1.0',
                'reason': 'Padrão para LSTM saudável',
                'effort': 'Médio (função de inicialização)'
            },
            {
                'priority': 4,
                'action': 'Monitoramento Específico LSTM',
                'change': 'Callback para monitorar gradientes LSTM',
                'reason': 'Detectar problemas precocemente',
                'effort': 'Médio (callback customizado)'
            }
        ]
        
        print("🎯 PRIORIDADE DE IMPLEMENTAÇÃO:")
        for action in immediate_actions:
            print(f"\n{action['priority']}. {action['action']}")
            print(f"   Mudança: {action['change']}")
            print(f"   Razão: {action['reason']}")
            print(f"   Esforço: {action['effort']}")
        
        return immediate_actions

def create_lstm_gradient_fix():
    """🔧 Criar correção específica para gradientes LSTM"""
    print("🔧 CRIANDO CORREÇÃO ESPECÍFICA PARA LSTM")
    print("=" * 60)
    
    fix_code = '''
def fix_lstm_gradients(model):
    """🔧 Aplicar correções nos gradientes LSTM"""
    
    # 1. Reinicializar LSTM com bias correto
    for name, module in model.named_modules():
        if isinstance(module, nn.LSTM):
            for param_name, param in module.named_parameters():
                if 'bias' in param_name:
                    # Forget gate bias = 1.0 (padrão LSTM)
                    n = param.size(0)
                    param.data.zero_()
                    param.data[n//4:n//2].fill_(1.0)
                    print(f"✅ {name}.{param_name}: Forget gate bias = 1.0")
                
                elif 'weight_ih' in param_name:
                    # Xavier para input-hidden weights
                    nn.init.xavier_uniform_(param)
                    print(f"✅ {name}.{param_name}: Xavier initialization")
                
                elif 'weight_hh' in param_name:
                    # Orthogonal para hidden-hidden weights
                    nn.init.orthogonal_(param)
                    print(f"✅ {name}.{param_name}: Orthogonal initialization")
    
    # 2. Aplicar gradient clipping mais suave
    return 0.5  # Novo max_grad_norm
    '''
    
    print("💻 CÓDIGO DA CORREÇÃO:")
    print(fix_code)
    
    return fix_code

if __name__ == "__main__":
    optimizer = LSTMGradientOptimizer()
    
    # Analisar problemas
    problems = optimizer.analyze_lstm_problems()
    
    # Gerar soluções
    solutions = optimizer.generate_solutions()
    
    # Recomendar ações imediatas
    actions = optimizer.recommend_immediate_actions()
    
    # Criar correção
    fix_code = create_lstm_gradient_fix()
    
    print(f"\n" + "=" * 60)
    print("🎯 RESUMO EXECUTIVO")
    print("=" * 60)
    
    print("🚨 PROBLEMA: LSTM com 15.43% gradientes zero")
    print("🎯 CAUSA: Vanishing gradients + inicialização ruim")
    print("💡 SOLUÇÃO RÁPIDA: Gradient clipping + LR warmup")
    print("🚀 SOLUÇÃO COMPLETA: Reinicialização + LayerNorm")
    
    print(f"\n🔥 IMPLEMENTAR AGORA:")
    print(f"1. max_grad_norm = 0.5 (era 1.0)")
    print(f"2. LR warmup de 10%")
    print(f"3. Forget gate bias = 1.0")
    print(f"4. Monitoramento LSTM específico")
    
    print(f"\n💪 RESULTADO ESPERADO:")
    print(f"✅ Gradientes LSTM: 15.43% → <5% zeros")
    print(f"✅ Estabilidade melhorada")
    print(f"✅ Convergência mais rápida")
    print(f"✅ Performance de trading melhor")