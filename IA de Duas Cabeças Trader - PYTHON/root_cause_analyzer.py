#!/usr/bin/env python3
"""
🔍 ANÁLISE DA CAUSA RAIZ DOS ZEROS NOS ACTION/VALUE NETWORKS
Investigar POR QUE 50-53% dos gradientes estão sendo zerados
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class RootCauseAnalyzer:
    """🔍 Analisador de causa raiz para zeros em gradientes"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_gradient_zeros_root_cause(self, model):
        """🎯 Análise completa da causa raiz dos zeros"""
        print("🔍 ANÁLISE DA CAUSA RAIZ - ZEROS NOS GRADIENTES")
        print("=" * 70)
        
        causes = {
            'dead_relu': 0,
            'vanishing_gradients': 0,
            'poor_initialization': 0,
            'learning_rate_issues': 0,
            'activation_saturation': 0,
            'weight_decay_issues': 0,
            'batch_norm_issues': 0,
            'architecture_problems': 0
        }
        
        # 1. Analisar ativações (Dead ReLU)
        dead_relu_analysis = self._analyze_dead_relu(model)
        causes['dead_relu'] = dead_relu_analysis['severity']
        
        # 2. Analisar vanishing gradients
        vanishing_analysis = self._analyze_vanishing_gradients(model)
        causes['vanishing_gradients'] = vanishing_analysis['severity']
        
        # 3. Analisar inicialização
        init_analysis = self._analyze_initialization_quality(model)
        causes['poor_initialization'] = init_analysis['severity']
        
        # 4. Analisar learning rate
        lr_analysis = self._analyze_learning_rate_impact(model)
        causes['learning_rate_issues'] = lr_analysis['severity']
        
        # 5. Analisar saturação de ativações
        saturation_analysis = self._analyze_activation_saturation(model)
        causes['activation_saturation'] = saturation_analysis['severity']
        
        # 6. Analisar arquitetura
        arch_analysis = self._analyze_architecture_problems(model)
        causes['architecture_problems'] = arch_analysis['severity']
        
        # Identificar causa principal
        main_cause = max(causes.items(), key=lambda x: x[1])
        
        print(f"\n🎯 CAUSA PRINCIPAL IDENTIFICADA:")
        print(f"   {main_cause[0].upper()}: {main_cause[1]:.1%} de impacto")
        
        # Gerar soluções baseadas na causa
        solutions = self._generate_root_cause_solutions(causes)
        
        return {
            'main_cause': main_cause[0],
            'severity': main_cause[1],
            'all_causes': causes,
            'solutions': solutions,
            'detailed_analysis': {
                'dead_relu': dead_relu_analysis,
                'vanishing_gradients': vanishing_analysis,
                'initialization': init_analysis,
                'learning_rate': lr_analysis,
                'saturation': saturation_analysis,
                'architecture': arch_analysis
            }
        }
    
    def _analyze_dead_relu(self, model) -> Dict:
        """🔍 Analisar Dead ReLU - principal suspeito"""
        print("\n🔍 ANALISANDO DEAD RELU...")
        
        dead_neurons = 0
        total_neurons = 0
        relu_layers = []
        
        # Simular forward pass para capturar ativações
        dummy_input = torch.randn(32, 1480)  # Batch de teste
        
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
            return hook
        
        # Registrar hooks em ReLUs
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
                relu_layers.append(name)
        
        # Forward pass
        try:
            with torch.no_grad():
                if hasattr(model, 'features_extractor'):
                    _ = model.features_extractor(dummy_input)
                else:
                    _ = model(dummy_input)
        except:
            print("⚠️ Erro no forward pass - usando análise alternativa")
        
        # Analisar ativações capturadas
        for name, activation in activations.items():
            if len(activation.shape) >= 2:
                # Contar neurônios mortos (sempre zero)
                dead_mask = torch.all(activation == 0, dim=0)
                dead_count = torch.sum(dead_mask).item()
                total_count = activation.shape[-1]
                
                dead_neurons += dead_count
                total_neurons += total_count
                
                dead_ratio = dead_count / total_count if total_count > 0 else 0
                print(f"   {name}: {dead_count}/{total_count} neurônios mortos ({dead_ratio:.1%})")
        
        # Limpar hooks
        for hook in hooks:
            hook.remove()
        
        overall_dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else 0
        
        print(f"📊 DEAD RELU TOTAL: {dead_neurons}/{total_neurons} ({overall_dead_ratio:.1%})")
        
        return {
            'severity': overall_dead_ratio,
            'dead_neurons': dead_neurons,
            'total_neurons': total_neurons,
            'affected_layers': len(relu_layers)
        }
    
    def _analyze_vanishing_gradients(self, model) -> Dict:
        """🔍 Analisar Vanishing Gradients"""
        print("\n🔍 ANALISANDO VANISHING GRADIENTS...")
        
        # Simular backward pass
        dummy_input = torch.randn(8, 1480, requires_grad=True)
        
        try:
            if hasattr(model, 'features_extractor'):
                output = model.features_extractor(dummy_input)
            else:
                output = model(dummy_input)
            
            loss = torch.mean(output ** 2)
            loss.backward()
            
            # Analisar magnitude dos gradientes por layer
            layer_grad_magnitudes = []
            very_small_grads = 0
            total_grads = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_magnitude = torch.norm(param.grad).item()
                    layer_grad_magnitudes.append((name, grad_magnitude))
                    
                    # Contar gradientes muito pequenos
                    small_grads = torch.sum(torch.abs(param.grad) < 1e-7).item()
                    very_small_grads += small_grads
                    total_grads += param.grad.numel()
                    
                    if grad_magnitude < 1e-6:
                        print(f"   ⚠️ {name}: gradiente muito pequeno ({grad_magnitude:.2e})")
            
            vanishing_ratio = very_small_grads / total_grads if total_grads > 0 else 0
            print(f"📊 VANISHING GRADIENTS: {very_small_grads}/{total_grads} ({vanishing_ratio:.1%})")
            
            return {
                'severity': vanishing_ratio,
                'very_small_grads': very_small_grads,
                'total_grads': total_grads,
                'layer_magnitudes': layer_grad_magnitudes[:5]  # Top 5
            }
            
        except Exception as e:
            print(f"⚠️ Erro na análise de vanishing gradients: {e}")
            return {'severity': 0.0, 'error': str(e)}
    
    def _analyze_initialization_quality(self, model) -> Dict:
        """🔍 Analisar qualidade da inicialização"""
        print("\n🔍 ANALISANDO QUALIDADE DA INICIALIZAÇÃO...")
        
        poor_init_layers = 0
        total_layers = 0
        init_issues = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                total_layers += 1
                param_data = param.data.cpu().numpy()
                
                # Verificar problemas de inicialização
                std = np.std(param_data)
                mean = np.mean(param_data)
                
                issues = []
                
                # 1. Desvio muito pequeno (pode causar vanishing)
                if std < 0.01:
                    issues.append(f"std muito baixo ({std:.4f})")
                
                # 2. Desvio muito grande (pode causar exploding)
                if std > 2.0:
                    issues.append(f"std muito alto ({std:.4f})")
                
                # 3. Média muito deslocada
                if abs(mean) > 0.1:
                    issues.append(f"média deslocada ({mean:.4f})")
                
                # 4. Distribuição não gaussiana
                if len(param_data.flatten()) > 100:
                    # Teste simples de normalidade
                    flat_data = param_data.flatten()
                    q75, q25 = np.percentile(flat_data, [75, 25])
                    iqr = q75 - q25
                    if iqr < std * 0.5:  # Distribuição muito concentrada
                        issues.append("distribuição suspeita")
                
                if issues:
                    poor_init_layers += 1
                    init_issues.append(f"{name}: {', '.join(issues)}")
                    print(f"   ⚠️ {name}: {', '.join(issues)}")
        
        poor_init_ratio = poor_init_layers / total_layers if total_layers > 0 else 0
        print(f"📊 INICIALIZAÇÃO PROBLEMÁTICA: {poor_init_layers}/{total_layers} ({poor_init_ratio:.1%})")
        
        return {
            'severity': poor_init_ratio,
            'poor_layers': poor_init_layers,
            'total_layers': total_layers,
            'issues': init_issues[:5]  # Top 5
        }
    
    def _analyze_learning_rate_impact(self, model) -> Dict:
        """🔍 Analisar impacto do learning rate"""
        print("\n🔍 ANALISANDO LEARNING RATE...")
        
        # Esta análise seria mais precisa com acesso ao optimizer
        # Por enquanto, análise baseada na magnitude dos pesos
        
        lr_issues = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_params += 1
                
                # Simular impacto do LR baseado na magnitude dos gradientes
                grad_norm = torch.norm(param.grad).item()
                param_norm = torch.norm(param.data).item()
                
                # Se gradientes são muito pequenos comparados aos pesos
                if param_norm > 0 and grad_norm / param_norm < 1e-6:
                    lr_issues += 1
                    print(f"   ⚠️ {name}: gradiente muito pequeno vs peso")
        
        lr_impact = lr_issues / total_params if total_params > 0 else 0
        print(f"📊 LEARNING RATE ISSUES: {lr_issues}/{total_params} ({lr_impact:.1%})")
        
        return {
            'severity': lr_impact,
            'affected_params': lr_issues,
            'total_params': total_params
        }
    
    def _analyze_activation_saturation(self, model) -> Dict:
        """🔍 Analisar saturação de ativações"""
        print("\n🔍 ANALISANDO SATURAÇÃO DE ATIVAÇÕES...")
        
        # Simular forward pass e capturar ativações
        dummy_input = torch.randn(16, 1480)
        saturated_layers = 0
        total_activation_layers = 0
        
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
            return hook
        
        # Registrar hooks em ativações
        for name, module in model.named_modules():
            if isinstance(module, (nn.Sigmoid, nn.Tanh, nn.ReLU)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
                total_activation_layers += 1
        
        try:
            with torch.no_grad():
                if hasattr(model, 'features_extractor'):
                    _ = model.features_extractor(dummy_input)
        except:
            pass
        
        # Analisar saturação
        for name, activation in activations.items():
            if 'sigmoid' in name.lower():
                # Sigmoid saturado: valores muito próximos de 0 ou 1
                saturated = torch.sum((activation > 0.99) | (activation < 0.01)).item()
                total = activation.numel()
                if saturated / total > 0.5:
                    saturated_layers += 1
                    print(f"   ⚠️ {name}: {saturated}/{total} saturados")
            
            elif 'tanh' in name.lower():
                # Tanh saturado: valores muito próximos de -1 ou 1
                saturated = torch.sum((activation > 0.99) | (activation < -0.99)).item()
                total = activation.numel()
                if saturated / total > 0.5:
                    saturated_layers += 1
                    print(f"   ⚠️ {name}: {saturated}/{total} saturados")
        
        # Limpar hooks
        for hook in hooks:
            hook.remove()
        
        saturation_ratio = saturated_layers / total_activation_layers if total_activation_layers > 0 else 0
        print(f"📊 SATURAÇÃO: {saturated_layers}/{total_activation_layers} ({saturation_ratio:.1%})")
        
        return {
            'severity': saturation_ratio,
            'saturated_layers': saturated_layers,
            'total_layers': total_activation_layers
        }
    
    def _analyze_architecture_problems(self, model) -> Dict:
        """🔍 Analisar problemas arquiteturais"""
        print("\n🔍 ANALISANDO PROBLEMAS ARQUITETURAIS...")
        
        arch_issues = []
        severity = 0.0
        
        # 1. Verificar depth excessiva
        max_depth = 0
        for name, _ in model.named_modules():
            depth = name.count('.')
            max_depth = max(max_depth, depth)
        
        if max_depth > 10:
            arch_issues.append(f"Rede muito profunda ({max_depth} níveis)")
            severity += 0.2
        
        # 2. Verificar bottlenecks
        layer_sizes = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_sizes.append((name, module.in_features, module.out_features))
        
        # Detectar bottlenecks severos
        for name, in_feat, out_feat in layer_sizes:
            if in_feat > out_feat * 10:  # Redução >90%
                arch_issues.append(f"Bottleneck severo: {name} ({in_feat}→{out_feat})")
                severity += 0.1
        
        # 3. Verificar skip connections ausentes
        has_residual = any('residual' in name.lower() or 'skip' in name.lower() 
                          for name, _ in model.named_modules())
        
        if not has_residual and max_depth > 5:
            arch_issues.append("Rede profunda sem skip connections")
            severity += 0.3
        
        print(f"📊 PROBLEMAS ARQUITETURAIS: {len(arch_issues)} detectados")
        for issue in arch_issues:
            print(f"   ⚠️ {issue}")
        
        return {
            'severity': min(severity, 1.0),
            'issues': arch_issues,
            'max_depth': max_depth,
            'has_residual': has_residual
        }
    
    def _generate_root_cause_solutions(self, causes: Dict) -> List[str]:
        """🎯 Gerar soluções baseadas na causa raiz"""
        solutions = []
        
        # Ordenar causas por severidade
        sorted_causes = sorted(causes.items(), key=lambda x: x[1], reverse=True)
        
        for cause, severity in sorted_causes:
            if severity > 0.3:  # Apenas causas significativas
                if cause == 'dead_relu':
                    solutions.extend([
                        "SUBSTITUIR ReLU por Leaky ReLU ou ELU",
                        "REDUZIR learning rate inicial",
                        "MELHORAR inicialização (He initialization)",
                        "ADICIONAR Batch Normalization antes das ativações"
                    ])
                
                elif cause == 'vanishing_gradients':
                    solutions.extend([
                        "ADICIONAR skip connections (ResNet-style)",
                        "USAR inicialização Xavier/Glorot",
                        "IMPLEMENTAR gradient clipping",
                        "REDUZIR profundidade da rede"
                    ])
                
                elif cause == 'poor_initialization':
                    solutions.extend([
                        "IMPLEMENTAR inicialização He para ReLU",
                        "USAR inicialização Xavier para Sigmoid/Tanh",
                        "APLICAR inicialização ortogonal para LSTM",
                        "VERIFICAR variance scaling"
                    ])
                
                elif cause == 'activation_saturation':
                    solutions.extend([
                        "SUBSTITUIR Sigmoid/Tanh por ReLU variants",
                        "ADICIONAR Batch Normalization",
                        "REDUZIR learning rate",
                        "USAR ativações mais suaves (Swish, GELU)"
                    ])
                
                elif cause == 'architecture_problems':
                    solutions.extend([
                        "ADICIONAR skip connections",
                        "REDUZIR profundidade da rede",
                        "EVITAR bottlenecks severos",
                        "IMPLEMENTAR attention mechanisms"
                    ])
        
        return list(set(solutions))  # Remove duplicatas

def analyze_daytrader_root_cause():
    """🔍 Analisar causa raiz no daytrader"""
    print("🔍 ANÁLISE DE CAUSA RAIZ - DAYTRADER")
    print("=" * 70)
    
    try:
        # Importar TwoHeadV6 (usada no daytrader)
        from trading_framework.policies.two_head_v6_intelligent_48h import TwoHeadV6Intelligent48h
        import gym
        from gym import spaces
        
        # Criar policy para análise
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1480,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)
        
        def lr_schedule(progress):
            return 3e-4
        
        policy = TwoHeadV6Intelligent48h(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            lstm_hidden_size=128
        )
        
        print("✅ TwoHeadV6 carregada para análise")
        
        # Executar análise de causa raiz
        analyzer = RootCauseAnalyzer()
        results = analyzer.analyze_gradient_zeros_root_cause(policy)
        
        # Exibir resultados
        print(f"\n" + "=" * 70)
        print("🎯 DIAGNÓSTICO FINAL")
        print("=" * 70)
        print(f"CAUSA PRINCIPAL: {results['main_cause'].upper()}")
        print(f"SEVERIDADE: {results['severity']:.1%}")
        
        print(f"\n📋 TODAS AS CAUSAS DETECTADAS:")
        for cause, severity in results['all_causes'].items():
            if severity > 0.1:
                print(f"   {cause}: {severity:.1%}")
        
        print(f"\n💡 SOLUÇÕES RECOMENDADAS:")
        for i, solution in enumerate(results['solutions'][:5], 1):
            print(f"   {i}. {solution}")
        
        return results
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = analyze_daytrader_root_cause()
    
    if results:
        print(f"\n" + "=" * 70)
        print("🎯 PRÓXIMOS PASSOS:")
        print("=" * 70)
        print("1. Implementar a solução para a CAUSA PRINCIPAL")
        print("2. Testar se os zeros diminuem")
        print("3. Aplicar soluções secundárias se necessário")
        print("4. Validar que o problema foi resolvido NA ORIGEM")
        print("\n💡 LEMBRE-SE: Corrigir a CAUSA, não os SINTOMAS!")
    else:
        print("\n❌ Análise falhou - revisar implementação")