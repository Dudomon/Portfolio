#!/usr/bin/env python3
"""
🔍 DEBUG SIGMOID SATURATION V7 - Investigar saturação dos sigmoids
Analisa onde e como os sigmoids estão saturando na arquitetura V7Intuition
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import RecurrentPPO
from datetime import datetime
import json

# Configurações
CHECKPOINT_PATH = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase3noisehandlingfixed_4900000_steps_20250814_112737.zip"
N_SAMPLES = 1000

def analyze_sigmoid_activations():
    """🔍 Análise detalhada das ativações sigmoid"""
    
    print("🔍 ANÁLISE DE SATURAÇÃO SIGMOID - V7 INTUITION")
    print("=" * 70)
    
    try:
        # Carregar modelo
        print("🤖 Carregando modelo...")
        model = RecurrentPPO.load(CHECKPOINT_PATH, device='cuda')
        model.policy.set_training_mode(False)  # Evaluation mode
        
        # Acessar policy V7
        policy = model.policy
        print(f"✅ Policy carregada: {type(policy).__name__}")
        
        # 1. IDENTIFICAR SIGMOIDS NA ARQUITETURA
        print("\n🧠 MAPEANDO SIGMOIDS NA ARQUITETURA:")
        sigmoid_layers = []
        
        # Verificar UnifiedBackbone gates
        if hasattr(policy, 'unified_backbone'):
            backbone = policy.unified_backbone
            
            if hasattr(backbone, 'actor_gate'):
                sigmoid_layers.append(('backbone.actor_gate', backbone.actor_gate))
                print("   ✅ Encontrado: backbone.actor_gate (Sigmoid)")
            
            if hasattr(backbone, 'critic_gate'): 
                sigmoid_layers.append(('backbone.critic_gate', backbone.critic_gate))
                print("   ✅ Encontrado: backbone.critic_gate (Sigmoid)")
        
        # Verificar outras camadas com sigmoid
        for name, module in policy.named_modules():
            if isinstance(module, torch.nn.Sigmoid):
                sigmoid_layers.append((name, module))
                print(f"   ✅ Encontrado: {name} (Sigmoid)")
        
        print(f"\n📊 Total de Sigmoids encontrados: {len(sigmoid_layers)}")
        
        if not sigmoid_layers:
            print("⚠️ Nenhum sigmoid encontrado na arquitetura!")
            return
        
        # 2. ANÁLISE DE SATURAÇÃO
        print(f"\n🔬 TESTANDO SATURAÇÃO COM {N_SAMPLES} SAMPLES...")
        
        # Resultados de saturação
        saturation_results = {}
        
        # Hook para capturar ativações
        activations = {}
        
        def create_hook(layer_name):
            def hook(module, input, output):
                # Capturar entrada e saída
                if isinstance(input, tuple) and len(input) > 0:
                    inp = input[0]
                else:
                    inp = input
                
                activations[layer_name] = {
                    'input': inp.detach().cpu().numpy() if hasattr(inp, 'detach') else inp,
                    'output': output.detach().cpu().numpy() if hasattr(output, 'detach') else output
                }
            return hook
        
        # Registrar hooks
        hooks = []
        for layer_name, layer_module in sigmoid_layers:
            hook = layer_module.register_forward_hook(create_hook(layer_name))
            hooks.append(hook)
        
        print("   📡 Hooks registrados")
        
        # Executar predições com observações variadas
        lstm_states = None
        all_raw_outputs = []
        all_sigmoid_outputs = []
        
        for i in range(N_SAMPLES):
            if i % 200 == 0:
                print(f"   📊 Sample {i+1}/{N_SAMPLES}...")
            
            # Observação aleatória (simulando dados reais)
            obs = np.random.normal(0, 1.0, (2580,)).astype(np.float32)
            
            # Predição
            try:
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                
                # Capturar outputs das ações (entry_quality está em action[1])
                if len(action) >= 2:
                    entry_quality = float(action[1])
                    all_sigmoid_outputs.append(entry_quality)
                
            except Exception as e:
                print(f"   ⚠️ Erro na predição {i}: {e}")
                continue
        
        # Remover hooks
        for hook in hooks:
            hook.remove()
        
        print(f"   ✅ Coletadas {len(all_sigmoid_outputs)} predições")
        
        # 3. ANÁLISE DAS ATIVAÇÕES DOS SIGMOIDS
        print(f"\n📊 ANÁLISE DAS ATIVAÇÕES SIGMOID:")
        
        for layer_name, _ in sigmoid_layers:
            if layer_name in activations:
                layer_data = activations[layer_name]
                
                # Análise da entrada (antes do sigmoid)
                inp = layer_data['input']
                out = layer_data['output']
                
                if hasattr(inp, '__len__') and len(inp) > 0:
                    inp_flat = inp.flatten() if hasattr(inp, 'flatten') else np.array(inp).flatten()
                    out_flat = out.flatten() if hasattr(out, 'flatten') else np.array(out).flatten()
                    
                    # Estatísticas da entrada
                    inp_mean = np.mean(inp_flat)
                    inp_std = np.std(inp_flat)
                    inp_min = np.min(inp_flat)
                    inp_max = np.max(inp_flat)
                    
                    # Estatísticas da saída
                    out_mean = np.mean(out_flat)
                    out_std = np.std(out_flat)
                    out_min = np.min(out_flat)
                    out_max = np.max(out_flat)
                    
                    # Detectar saturação
                    # Entrada: valores muito altos/baixos saturam sigmoid
                    saturated_high = np.sum(inp_flat > 5.0)  # sigmoid(5) ≈ 0.993
                    saturated_low = np.sum(inp_flat < -5.0)   # sigmoid(-5) ≈ 0.007
                    total_values = len(inp_flat)
                    
                    # Saída: valores próximos de 0 ou 1
                    output_near_zero = np.sum(out_flat < 0.1)
                    output_near_one = np.sum(out_flat > 0.9)
                    
                    saturation_pct = (saturated_high + saturated_low) / total_values * 100
                    extreme_outputs_pct = (output_near_zero + output_near_one) / total_values * 100
                    
                    print(f"\n   🔍 {layer_name}:")
                    print(f"     📊 Entrada: μ={inp_mean:.3f}, σ={inp_std:.3f}, range=[{inp_min:.3f}, {inp_max:.3f}]")
                    print(f"     📊 Saída:   μ={out_mean:.3f}, σ={out_std:.3f}, range=[{out_min:.3f}, {out_max:.3f}]")
                    print(f"     🚨 Saturação entrada: {saturation_pct:.1f}% (>{saturated_high}, <{saturated_low})")
                    print(f"     🚨 Saídas extremas: {extreme_outputs_pct:.1f}% (near 0: {output_near_zero}, near 1: {output_near_one})")
                    
                    # Classificar severidade
                    if saturation_pct > 70:
                        severity = "🔥 CRÍTICA"
                    elif saturation_pct > 40:
                        severity = "⚠️ ALTA"
                    elif saturation_pct > 20:
                        severity = "🟡 MODERADA"
                    else:
                        severity = "✅ BAIXA"
                    
                    print(f"     🎯 Severidade saturação: {severity}")
                    
                    # Armazenar resultados
                    saturation_results[layer_name] = {
                        'input_stats': {'mean': inp_mean, 'std': inp_std, 'min': inp_min, 'max': inp_max},
                        'output_stats': {'mean': out_mean, 'std': out_std, 'min': out_min, 'max': out_max},
                        'saturation_pct': saturation_pct,
                        'extreme_outputs_pct': extreme_outputs_pct,
                        'severity': severity,
                        'saturated_high': int(saturated_high),
                        'saturated_low': int(saturated_low),
                        'output_near_zero': int(output_near_zero),
                        'output_near_one': int(output_near_one)
                    }
        
        # 4. ANÁLISE ENTRY QUALITY ESPECÍFICA
        print(f"\n🎯 ANÁLISE ENTRY QUALITY (ACTION[1]):")
        if all_sigmoid_outputs:
            entry_qualities = np.array(all_sigmoid_outputs)
            
            eq_mean = np.mean(entry_qualities)
            eq_std = np.std(entry_qualities)
            eq_min = np.min(entry_qualities)
            eq_max = np.max(entry_qualities)
            
            # Concentração em extremos (0 e 1)
            eq_near_zero = np.sum(entry_qualities < 0.1)
            eq_near_one = np.sum(entry_qualities > 0.9)
            eq_extremes_pct = (eq_near_zero + eq_near_one) / len(entry_qualities) * 100
            
            print(f"   📊 Entry Quality: μ={eq_mean:.3f}, σ={eq_std:.3f}, range=[{eq_min:.3f}, {eq_max:.3f}]")
            print(f"   🚨 Extremos: {eq_extremes_pct:.1f}% (0s: {eq_near_zero}, 1s: {eq_near_one})")
            
            # Distribuição detalhada
            hist, bins = np.histogram(entry_qualities, bins=10, range=(0, 1))
            print(f"   📈 Distribuição por bins:")
            for i, (count, bin_start) in enumerate(zip(hist, bins[:-1])):
                bin_end = bins[i+1]
                pct = (count / len(entry_qualities)) * 100
                bar = "█" * max(1, int(pct / 2))
                print(f"     [{bin_start:.1f}-{bin_end:.1f}]: {count:4d} ({pct:5.1f}%) {bar}")
        
        # 5. DIAGNÓSTICO E RECOMENDAÇÕES
        print(f"\n🔬 DIAGNÓSTICO GERAL:")
        
        # Verificar se há saturação crítica
        critical_layers = [name for name, result in saturation_results.items() 
                          if result['saturation_pct'] > 70]
        high_saturation_layers = [name for name, result in saturation_results.items() 
                                 if 40 < result['saturation_pct'] <= 70]
        
        if critical_layers:
            print(f"   🔥 SATURAÇÃO CRÍTICA detectada em: {critical_layers}")
            print(f"   💡 AÇÃO IMEDIATA: Reduzir learning rate ou modificar inicialização")
        elif high_saturation_layers:
            print(f"   ⚠️ SATURAÇÃO ALTA em: {high_saturation_layers}")
            print(f"   💡 RECOMENDAÇÃO: Monitorar e considerar ajustes")
        else:
            print(f"   ✅ Saturação dentro dos limites aceitáveis")
        
        # Verificar Entry Quality especificamente
        if eq_extremes_pct > 90:
            print(f"   🚨 ENTRY QUALITY SATURADA: {eq_extremes_pct:.1f}% em extremos")
            print(f"   💡 SOLUÇÃO: Usar clipping ao invés de sigmoid")
            print(f"   💡 CÓDIGO: actions[:, 1] = torch.clamp((raw + 1.0) / 2.0, 0.0, 1.0)")
        
        # 6. SALVAR RELATÓRIO
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"D:/Projeto/avaliacoes/sigmoid_saturation_analysis_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'checkpoint': CHECKPOINT_PATH.split('/')[-1],
            'samples_analyzed': N_SAMPLES,
            'sigmoid_layers_found': len(sigmoid_layers),
            'saturation_results': saturation_results,
            'entry_quality_analysis': {
                'mean': float(eq_mean) if 'eq_mean' in locals() else 0,
                'std': float(eq_std) if 'eq_std' in locals() else 0,
                'extremes_pct': float(eq_extremes_pct) if 'eq_extremes_pct' in locals() else 0,
                'distribution': hist.tolist() if 'hist' in locals() else []
            },
            'critical_layers': critical_layers,
            'high_saturation_layers': high_saturation_layers,
            'diagnosis': {
                'status': 'critical' if critical_layers else 'high' if high_saturation_layers else 'normal',
                'recommendation': 'immediate_action' if critical_layers else 'monitor' if high_saturation_layers else 'continue'
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Relatório salvo: {report_file}")
        
        # 7. RESUMO EXECUTIVO
        print(f"\n📋 RESUMO EXECUTIVO:")
        print(f"   🔍 Sigmoids analisados: {len(sigmoid_layers)}")
        print(f"   🚨 Entry Quality extremos: {eq_extremes_pct:.1f}%" if 'eq_extremes_pct' in locals() else "   🚨 Entry Quality: Não analisado")
        print(f"   🔥 Camadas críticas: {len(critical_layers)}")
        print(f"   ⚠️ Camadas com alta saturação: {len(high_saturation_layers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"🚀 INICIANDO ANÁLISE SIGMOID - {datetime.now().strftime('%H:%M:%S')}")
    
    success = analyze_sigmoid_activations()
    
    if success:
        print(f"\n✅ ANÁLISE CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")
    else:
        print(f"\n❌ ANÁLISE FALHOU - {datetime.now().strftime('%H:%M:%S')}")