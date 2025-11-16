#!/usr/bin/env python3
"""
🔍 DEBUG INDIVIDUAL SIGMOID LAYERS - Análise detalhada de cada camada
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from datetime import datetime

CHECKPOINT_PATH = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase4integration_6200000_steps_20250813_161212.zip"

def analyze_individual_sigmoids():
    """Análise detalhada de cada sigmoid individual"""
    
    print("🔍 ANÁLISE INDIVIDUAL SIGMOIDS - V7 INTUITION")
    print("=" * 60)
    
    try:
        # Carregar modelo
        print("🤖 Carregando modelo...")
        model = RecurrentPPO.load(CHECKPOINT_PATH, device='cuda')
        policy = model.policy
        
        # Mapear todos os sigmoids
        sigmoid_modules = {}
        
        print("\n🧠 MAPEANDO SIGMOIDS:")
        for name, module in policy.named_modules():
            if isinstance(module, torch.nn.Sigmoid):
                sigmoid_modules[name] = module
                print(f"   ✅ {name}")
        
        print(f"\n📊 Total: {len(sigmoid_modules)} sigmoids encontrados")
        
        if not sigmoid_modules:
            print("❌ Nenhum sigmoid encontrado!")
            return
        
        # Preparar coleta de ativações
        all_activations = {}
        
        def create_activation_hook(layer_name):
            def hook(module, input, output):
                # Capturar apenas a saída (pós-sigmoid)
                if hasattr(output, 'detach'):
                    activation = output.detach().cpu().numpy()
                    if layer_name not in all_activations:
                        all_activations[layer_name] = []
                    all_activations[layer_name].append(activation)
            return hook
        
        # Registrar hooks
        hooks = []
        for layer_name, sigmoid_module in sigmoid_modules.items():
            hook = sigmoid_module.register_forward_hook(create_activation_hook(layer_name))
            hooks.append(hook)
        
        print("\n🔬 COLETANDO ATIVAÇÕES (100 samples)...")
        
        # Executar predições
        lstm_states = None
        for i in range(100):  # Menos samples para teste mais rápido
            if i % 25 == 0:
                print(f"   📊 Sample {i+1}/100...")
            
            # Observação aleatória
            obs = np.random.normal(0, 1.0, (2580,)).astype(np.float32)
            
            try:
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
            except Exception as e:
                print(f"   ⚠️ Erro na predição {i}: {e}")
                continue
        
        # Remover hooks
        for hook in hooks:
            hook.remove()
        
        print(f"\n📊 ANÁLISE DAS ATIVAÇÕES:")
        
        # Analisar cada sigmoid
        saturation_summary = {}
        
        for layer_name in sigmoid_modules.keys():
            if layer_name in all_activations and len(all_activations[layer_name]) > 0:
                
                # Concatenar todas as ativações
                layer_activations = []
                for activation_batch in all_activations[layer_name]:
                    if hasattr(activation_batch, 'flatten'):
                        layer_activations.extend(activation_batch.flatten())
                    else:
                        layer_activations.extend(np.array(activation_batch).flatten())
                
                if len(layer_activations) == 0:
                    continue
                
                activations_array = np.array(layer_activations)
                
                # Estatísticas
                mean_val = np.mean(activations_array)
                std_val = np.std(activations_array)
                min_val = np.min(activations_array)
                max_val = np.max(activations_array)
                
                # Saturação (valores extremos)
                near_zero = np.sum(activations_array < 0.1)
                near_one = np.sum(activations_array > 0.9)
                total_values = len(activations_array)
                saturation_pct = (near_zero + near_one) / total_values * 100
                
                # Classificar severidade
                if saturation_pct > 80:
                    severity = "🔥 CRÍTICA"
                    risk_level = "CRITICAL"
                elif saturation_pct > 60:
                    severity = "⚠️ ALTA"
                    risk_level = "HIGH"
                elif saturation_pct > 40:
                    severity = "🟡 MODERADA"
                    risk_level = "MODERATE"
                else:
                    severity = "✅ BAIXA"
                    risk_level = "LOW"
                
                print(f"\n🔍 {layer_name}:")
                print(f"   📊 Stats: μ={mean_val:.3f}, σ={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
                print(f"   🚨 Saturação: {saturation_pct:.1f}% (0s: {near_zero}, 1s: {near_one})")
                print(f"   🎯 Severidade: {severity}")
                
                saturation_summary[layer_name] = {
                    'saturation_pct': saturation_pct,
                    'severity': severity,
                    'risk_level': risk_level,
                    'near_zero': near_zero,
                    'near_one': near_one,
                    'total_values': total_values,
                    'stats': {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val
                    }
                }
        
        # RESUMO E RECOMENDAÇÕES
        print(f"\n🔬 ANÁLISE GERAL:")
        
        critical_layers = [name for name, data in saturation_summary.items() 
                          if data['risk_level'] == 'CRITICAL']
        high_risk_layers = [name for name, data in saturation_summary.items() 
                           if data['risk_level'] == 'HIGH']
        moderate_risk_layers = [name for name, data in saturation_summary.items() 
                               if data['risk_level'] == 'MODERATE']
        
        print(f"   🔥 Críticas: {len(critical_layers)}")
        print(f"   ⚠️ Alto risco: {len(high_risk_layers)}")
        print(f"   🟡 Risco moderado: {len(moderate_risk_layers)}")
        print(f"   ✅ Baixo risco: {len(saturation_summary) - len(critical_layers) - len(high_risk_layers) - len(moderate_risk_layers)}")
        
        # Identificar quais sigmoids substituir
        sigmoids_to_replace = []
        
        if critical_layers:
            print(f"\n🚨 SIGMOIDS CRÍTICOS (>80% saturação):")
            for layer in critical_layers:
                print(f"   🔥 {layer}: {saturation_summary[layer]['saturation_pct']:.1f}%")
                sigmoids_to_replace.append(layer)
        
        if high_risk_layers:
            print(f"\n⚠️ SIGMOIDS ALTO RISCO (60-80% saturação):")
            for layer in high_risk_layers:
                print(f"   ⚠️ {layer}: {saturation_summary[layer]['saturation_pct']:.1f}%")
                sigmoids_to_replace.append(layer)
        
        # RECOMENDAÇÕES FINAIS
        print(f"\n💡 RECOMENDAÇÕES:")
        
        total_problematic = len(critical_layers) + len(high_risk_layers)
        
        if total_problematic == 0:
            print(f"   ✅ Nenhuma saturação crítica detectada")
            print(f"   💡 Pode continuar treinamento atual")
        elif total_problematic <= 3:
            print(f"   🔧 {total_problematic} sigmoids problemáticos - FIX SELETIVO")
            print(f"   💡 Aplicar tanh apenas nas camadas problemáticas")
            print(f"   💡 Pode tentar warm restart com LR baixo")
        else:
            print(f"   🔥 {total_problematic} sigmoids problemáticos - FIX GLOBAL")
            print(f"   💡 SUBSTITUIR TODOS os sigmoids por tanh")
            print(f"   💡 REINICIAR treinamento do zero (recomendado)")
        
        print(f"\n🎯 SIGMOIDS PARA SUBSTITUIR:")
        if sigmoids_to_replace:
            for layer in sigmoids_to_replace:
                print(f"   🔧 {layer}")
        else:
            print(f"   ✅ Nenhum sigmoid precisa ser substituído urgentemente")
        
        # Salvar relatório detalhado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"D:/Projeto/avaliacoes/individual_sigmoid_analysis_{timestamp}.json"
        
        import json
        report = {
            'timestamp': timestamp,
            'checkpoint': CHECKPOINT_PATH.split('/')[-1],
            'total_sigmoids': len(sigmoid_modules),
            'analyzed_sigmoids': len(saturation_summary),
            'saturation_details': saturation_summary,
            'risk_summary': {
                'critical': len(critical_layers),
                'high_risk': len(high_risk_layers),
                'moderate_risk': len(moderate_risk_layers),
                'low_risk': len(saturation_summary) - total_problematic - len(moderate_risk_layers)
            },
            'sigmoids_to_replace': sigmoids_to_replace,
            'recommendation': 'global_fix' if total_problematic > 3 else 'selective_fix' if total_problematic > 0 else 'continue'
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Relatório detalhado salvo: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"🚀 INICIANDO ANÁLISE INDIVIDUAL - {datetime.now().strftime('%H:%M:%S')}")
    
    success = analyze_individual_sigmoids()
    
    if success:
        print(f"\n✅ ANÁLISE CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")
    else:
        print(f"\n❌ ANÁLISE FALHOU - {datetime.now().strftime('%H:%M:%S')}")