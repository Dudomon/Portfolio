#!/usr/bin/env python3
"""
🎯 AVALIAÇÃO V7 CHECKPOINT 4M - Performance + Monitor de Saturação
Teste completo do checkpoint de 4M steps do daytrader
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import torch
import pandas as pd
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("🎯 AVALIAÇÃO V7 CHECKPOINT 4M - SISTEMA COMPLETO")
print("=" * 80)

def create_saturation_monitor():
    """Criar monitor de saturação para análise de gradientes"""
    class SaturationMonitor:
        def __init__(self):
            self.saturation_data = []
            self.gradient_data = []
            self.activation_data = []
            
        def analyze_model_saturation(self, model):
            """Analisar saturação do modelo"""
            saturation_report = {
                'timestamp': datetime.now().isoformat(),
                'total_params': 0,
                'zero_params': 0,
                'saturated_params': 0,
                'components': {}
            }
            
            for name, param in model.policy.named_parameters():
                if param.numel() == 0:
                    continue
                    
                param_data = param.detach().cpu().numpy()
                total_elements = param.numel()
                
                # Análise de zeros
                zero_mask = np.abs(param_data) < 1e-8
                zero_count = np.sum(zero_mask)
                
                # Análise de saturação (valores muito próximos dos extremos)
                if 'tanh' in name.lower() or 'sigmoid' in name.lower():
                    # Para ativações Tanh/Sigmoid
                    saturated_mask = (np.abs(param_data) > 0.95)
                else:
                    # Para weights normais
                    saturated_mask = (np.abs(param_data) > 3.0)
                
                saturated_count = np.sum(saturated_mask)
                
                component_report = {
                    'total_elements': total_elements,
                    'zero_count': zero_count,
                    'zero_ratio': zero_count / total_elements,
                    'saturated_count': saturated_count,
                    'saturated_ratio': saturated_count / total_elements,
                    'mean': float(np.mean(param_data)),
                    'std': float(np.std(param_data)),
                    'min': float(np.min(param_data)),
                    'max': float(np.max(param_data))
                }
                
                saturation_report['components'][name] = component_report
                saturation_report['total_params'] += total_elements
                saturation_report['zero_params'] += zero_count
                saturation_report['saturated_params'] += saturated_count
            
            # Calcular ratios globais
            if saturation_report['total_params'] > 0:
                saturation_report['global_zero_ratio'] = saturation_report['zero_params'] / saturation_report['total_params']
                saturation_report['global_saturated_ratio'] = saturation_report['saturated_params'] / saturation_report['total_params']
            
            self.saturation_data.append(saturation_report)
            return saturation_report
            
        def analyze_gradients(self, model):
            """Analisar gradientes durante treinamento"""
            gradient_report = {
                'timestamp': datetime.now().isoformat(),
                'components': {}
            }
            
            for name, param in model.policy.named_parameters():
                if param.grad is not None and param.numel() > 0:
                    grad_data = param.grad.detach().cpu().numpy()
                    
                    gradient_report['components'][name] = {
                        'mean': float(np.mean(grad_data)),
                        'std': float(np.std(grad_data)),
                        'max': float(np.max(np.abs(grad_data))),
                        'zero_ratio': float(np.sum(np.abs(grad_data) < 1e-8) / grad_data.size)
                    }
            
            self.gradient_data.append(gradient_report)
            return gradient_report
            
        def generate_report(self):
            """Gerar relatório completo"""
            if not self.saturation_data:
                return "Nenhum dado de saturação coletado"
            
            latest = self.saturation_data[-1]
            
            report = f"""
🎯 RELATÓRIO DE SATURAÇÃO - {latest['timestamp']}
{'='*60}

📊 ESTATÍSTICAS GLOBAIS:
   Total de Parâmetros: {latest['total_params']:,}
   Parâmetros Zerados: {latest['zero_params']:,} ({latest.get('global_zero_ratio', 0)*100:.2f}%)
   Parâmetros Saturados: {latest['saturated_params']:,} ({latest.get('global_saturated_ratio', 0)*100:.2f}%)

🔍 COMPONENTES CRÍTICOS:
"""
            
            # Análise por componente
            critical_components = []
            for name, data in latest['components'].items():
                if data['zero_ratio'] > 0.1 or data['saturated_ratio'] > 0.1:
                    critical_components.append((name, data))
            
            if critical_components:
                for name, data in critical_components[:10]:  # Top 10
                    report += f"""
   {name}:
      Zeros: {data['zero_ratio']*100:.1f}% | Saturados: {data['saturated_ratio']*100:.1f}%
      Range: [{data['min']:.3f}, {data['max']:.3f}] | Std: {data['std']:.3f}
"""
            else:
                report += "   ✅ Nenhum componente crítico detectado\n"
            
            return report
    
    return SaturationMonitor()

def load_checkpoint_and_evaluate():
    """Carregar checkpoint e executar avaliação completa"""
    
    try:
        # 1. Carregar checkpoint
        print("1. 📂 CARREGANDO CHECKPOINT 4M:")
        checkpoint_path = "./Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase2riskmanagement_4000000_steps_20250814_093028.zip"
        
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "./trading_framework/training/checkpoints/DAYTRADER/checkpoint_4000000_steps_20250814_093028.zip"
        
        if not os.path.exists(checkpoint_path):
            print(f"   ❌ Checkpoint não encontrado: {checkpoint_path}")
            return
        
        print(f"   ✅ Checkpoint encontrado: {checkpoint_path}")
        
        # Import necessário
        from sb3_contrib import RecurrentPPO
        from trading_framework.policies.two_head_v7_simple import TwoHeadV7Simple
        
        # Carregar modelo
        print("   📦 Carregando modelo...")
        model = RecurrentPPO.load(checkpoint_path)
        print(f"   ✅ Modelo carregado: {type(model).__name__}")
        print(f"   ✅ Política: {type(model.policy).__name__}")
        print(f"   ✅ Device: {model.device}")
        print(f"   ✅ Steps treinados: {getattr(model, 'num_timesteps', 'unknown')}")
        
        # 2. Criar monitor de saturação
        print("\n2. 🔍 CRIANDO MONITOR DE SATURAÇÃO:")
        saturation_monitor = create_saturation_monitor()
        print("   ✅ Monitor de saturação criado")
        
        # 3. Análise inicial de saturação
        print("\n3. 📊 ANÁLISE INICIAL DE SATURAÇÃO:")
        saturation_report = saturation_monitor.analyze_model_saturation(model)
        print(f"   Total parâmetros: {saturation_report['total_params']:,}")
        print(f"   Parâmetros zerados: {saturation_report['zero_params']:,} ({saturation_report.get('global_zero_ratio', 0)*100:.2f}%)")
        print(f"   Parâmetros saturados: {saturation_report['saturated_params']:,} ({saturation_report.get('global_saturated_ratio', 0)*100:.2f}%)")
        
        # 4. Análise detalhada por componente
        print("\n4. 🔍 ANÁLISE DETALHADA POR COMPONENTE:")
        
        # Features Extractor
        fe_components = {k: v for k, v in saturation_report['components'].items() if 'features_extractor' in k}
        if fe_components:
            print("   📊 FEATURES EXTRACTOR:")
            for name, data in list(fe_components.items())[:5]:
                print(f"      {name.split('.')[-2:]}: zeros={data['zero_ratio']*100:.1f}%, sat={data['saturated_ratio']*100:.1f}%")
        
        # Actor components
        actor_components = {k: v for k, v in saturation_report['components'].items() if 'actor' in k.lower()}
        if actor_components:
            print("   🎭 ACTOR:")
            for name, data in list(actor_components.items())[:5]:
                print(f"      {name.split('.')[-1]}: zeros={data['zero_ratio']*100:.1f}%, sat={data['saturated_ratio']*100:.1f}%")
        
        # Critic components
        critic_components = {k: v for k, v in saturation_report['components'].items() if 'critic' in k.lower()}
        if critic_components:
            print("   🎯 CRITIC:")
            for name, data in list(critic_components.items())[:5]:
                print(f"      {name.split('.')[-1]}: zeros={data['zero_ratio']*100:.1f}%, sat={data['saturated_ratio']*100:.1f}%")
        
        # 5. Teste de performance (sem environment - apenas análise do modelo)
        print("\n5. ⚡ TESTE DE PERFORMANCE DO MODELO:")
        
        # Criar dados de teste sintéticos (simulando observações de trading)
        batch_size = 32
        features_dim = 256  # Baseado na configuração V7
        
        # Simular observações
        dummy_obs = torch.randn(batch_size, features_dim, device=model.device)
        
        # Simular LSTM states
        lstm_states = (
            torch.zeros(1, batch_size, 256, device=model.device),  # h
            torch.zeros(1, batch_size, 256, device=model.device)   # c
        )
        episode_starts = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
        
        # Teste de throughput
        print("   🚀 Teste de throughput:")
        num_iterations = 100
        
        model.policy.eval()
        with torch.no_grad():
            start_time = time.time()
            
            for i in range(num_iterations):
                try:
                    # Forward pass completo
                    actions, values, log_probs, new_lstm_states = model.policy.forward(
                        dummy_obs, lstm_states, episode_starts
                    )
                    lstm_states = new_lstm_states
                except Exception as e:
                    print(f"      ⚠️ Erro no forward pass {i}: {e}")
                    break
            
            end_time = time.time()
            
            if i > 0:
                total_time = end_time - start_time
                throughput = (i * batch_size) / total_time
                print(f"      ✅ Throughput: {throughput:.1f} inferences/sec")
                print(f"      ✅ Latência média: {(total_time/i)*1000:.1f}ms per batch")
            
        # 6. Análise de outputs
        print("\n6. 📈 ANÁLISE DE OUTPUTS:")
        if 'actions' in locals():
            print(f"   Actions shape: {actions.shape}")
            print(f"   Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"   Actions mean: {actions.mean():.3f} ± {actions.std():.3f}")
            
        if 'values' in locals():
            print(f"   Values shape: {values.shape}")
            print(f"   Values range: [{values.min():.3f}, {values.max():.3f}]")
            print(f"   Values mean: {values.mean():.3f} ± {values.std():.3f}")
        
        # 7. Relatório final
        print("\n7. 📋 RELATÓRIO FINAL:")
        final_report = saturation_monitor.generate_report()
        print(final_report)
        
        # 8. Salvar relatório
        print("\n8. 💾 SALVANDO RELATÓRIO:")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"avaliacoes/avaliacao_v7_4m_{timestamp}.txt"
        
        os.makedirs("avaliacoes", exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"AVALIAÇÃO V7 CHECKPOINT 4M - {timestamp}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Modelo: {type(model).__name__}\n")
            f.write(f"Política: {type(model.policy).__name__}\n")
            f.write(f"Device: {model.device}\n")
            f.write(f"Steps: {getattr(model, 'num_timesteps', 'unknown')}\n\n")
            f.write(final_report)
            
            if 'throughput' in locals():
                f.write(f"\n\nPERFORMANCE:\n")
                f.write(f"Throughput: {throughput:.1f} inferences/sec\n")
                f.write(f"Latência: {(total_time/i)*1000:.1f}ms per batch\n")
        
        print(f"   ✅ Relatório salvo: {report_file}")
        
        return {
            'saturation_report': saturation_report,
            'throughput': locals().get('throughput', 0),
            'model_info': {
                'type': type(model).__name__,
                'policy': type(model.policy).__name__,
                'device': str(model.device),
                'steps': getattr(model, 'num_timesteps', 'unknown')
            }
        }
        
    except Exception as e:
        print(f"❌ ERRO na avaliação: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    start_time = time.time()
    
    result = load_checkpoint_and_evaluate()
    
    end_time = time.time()
    
    print(f"\n🎯 AVALIAÇÃO CONCLUÍDA em {end_time - start_time:.1f}s")
    
    if result:
        print("✅ Resultado disponível para análise")
        
        # Summary
        saturation = result['saturation_report']
        print(f"\n📊 RESUMO:")
        print(f"   Parâmetros totais: {saturation['total_params']:,}")
        print(f"   Zero ratio: {saturation.get('global_zero_ratio', 0)*100:.2f}%")
        print(f"   Saturated ratio: {saturation.get('global_saturated_ratio', 0)*100:.2f}%")
        if 'throughput' in result:
            print(f"   Performance: {result['throughput']:.1f} inferences/sec")
    else:
        print("❌ Avaliação falhou")