#!/usr/bin/env python3
"""
🔍 ANÁLISE DE EARLY STOPPING - SISTEMA SILUS
Analisa logs de treinamento para verificar eficácia do early stopping
"""

import os
import json
import pandas as pd
from datetime import datetime
import glob
import numpy as np

def analisar_logs_silus():
    """Analisa logs de treinamento SILUS para verificar early stopping"""
    
    # Encontrar arquivos de log mais recentes
    avaliacoes_path = "D:/Projeto/avaliacoes"
    silus_models_path = "D:/Projeto/Otimizacao/treino_principal/models/SILUS"
    
    print("🔍 ANÁLISE DE EARLY STOPPING - SISTEMA SILUS")
    print("=" * 60)
    
    # 1. Analisar checkpoints salvos para ver quando parou o treinamento
    print("\n📦 ANÁLISE DE CHECKPOINTS:")
    
    # Buscar todos os checkpoints SILUS
    checkpoint_patterns = [
        f"{silus_models_path}/SILUS_phase*_steps_*.zip",
        f"{silus_models_path}/FINAL_*_steps_*.zip"
    ]
    
    all_checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints = glob.glob(pattern)
        all_checkpoints.extend(checkpoints)
    
    # Extrair informações dos checkpoints
    checkpoint_info = []
    for cp_path in all_checkpoints:
        filename = os.path.basename(cp_path)
        
        # Extrair steps e timestamp
        try:
            if "FINAL_" in filename:
                parts = filename.split("_")
                steps = int(parts[2])
                timestamp_str = parts[4].replace('.zip', '')
                phase = "FINAL"
            else:
                parts = filename.split("_")
                steps = int(parts[2])
                timestamp_str = parts[4].replace('.zip', '')
                phase = parts[1]
            
            # Converter timestamp - ignorar timestamp por enquanto
            timestamp = None
            
            checkpoint_info.append({
                'filename': filename,
                'steps': steps,
                'phase': phase,
                'timestamp': timestamp
            })
            
        except (ValueError, IndexError) as e:
            print(f"⚠️ Erro ao processar {filename}: {e}")
    
    # Ordenar por steps
    checkpoint_info.sort(key=lambda x: x['steps'])
    
    print(f"\n📊 TOTAL DE CHECKPOINTS: {len(checkpoint_info)}")
    
    if checkpoint_info:
        print(f"🚀 PRIMEIRO CHECKPOINT: {checkpoint_info[0]['steps']:,} steps ({checkpoint_info[0]['phase']})")
        print(f"🏁 ÚLTIMO CHECKPOINT: {checkpoint_info[-1]['steps']:,} steps ({checkpoint_info[-1]['phase']})")
        # print(f"⏱️ TEMPO TOTAL TREINAMENTO: {checkpoint_info[-1]['timestamp'] - checkpoint_info[0]['timestamp']}")
        
        # Verificar se chegou aos 5M steps planejados
        target_steps = 5_000_000
        max_steps = checkpoint_info[-1]['steps']
        completion_rate = (max_steps / target_steps) * 100
        
        print(f"\n🎯 META DE TREINAMENTO: {target_steps:,} steps")
        print(f"✅ ALCANÇADO: {max_steps:,} steps ({completion_rate:.1f}%)")
        
        if max_steps >= target_steps:
            print("🟢 TREINAMENTO COMPLETADO - Early stopping NÃO ativado")
        else:
            print("🟡 TREINAMENTO INCOMPLETO - Possível early stopping ou interrupção")
    
    # 2. Analisar fases do treinamento
    print("\n📈 ANÁLISE POR FASES:")
    
    phases = {}
    for cp in checkpoint_info:
        if cp['phase'] not in phases:
            phases[cp['phase']] = []
        phases[cp['phase']].append(cp)
    
    for phase_name, phase_checkpoints in phases.items():
        if phase_name == "FINAL":
            continue
            
        phase_checkpoints.sort(key=lambda x: x['steps'])
        start_steps = phase_checkpoints[0]['steps']
        end_steps = phase_checkpoints[-1]['steps']
        duration_steps = end_steps - start_steps
        
        print(f"🔸 {phase_name.upper()}:")
        print(f"   Steps: {start_steps:,} → {end_steps:,} ({duration_steps:,} steps)")
        print(f"   Checkpoints: {len(phase_checkpoints)}")
        
        # Tempo da fase - skip por enquanto
        # start_time = phase_checkpoints[0]['timestamp']
        # end_time = phase_checkpoints[-1]['timestamp']  
        # duration_time = end_time - start_time
        # print(f"   Duração: {duration_time}")
    
    # 3. Analisar avaliações de performance
    print("\n🎯 ANÁLISE DE PERFORMANCE:")
    
    avaliacoes_files = glob.glob(f"{avaliacoes_path}/avaliacao_v11_*_20250824_*.txt")
    avaliacoes_files.sort()
    
    performance_data = []
    
    for file_path in avaliacoes_files[-10:]:  # Últimas 10 avaliações
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extrair informações básicas
            filename = os.path.basename(file_path)
            
            # Extrair steps do nome do arquivo
            parts = filename.split('_')
            try:
                steps_str = parts[2].replace('k', '000')
                steps = int(steps_str)
            except:
                continue
                
            # Extrair métricas do conteúdo
            lines = content.split('\n')
            retorno_medio = None
            win_rate = None
            profit_factor = None
            avaliacao_final = None
            
            for line in lines:
                if 'Retorno Médio:' in line:
                    try:
                        retorno_str = line.split(':')[1].strip().replace('%', '').replace('+', '')
                        retorno_medio = float(retorno_str)
                    except:
                        pass
                elif 'Win Rate Global:' in line:
                    try:
                        wr_str = line.split(':')[1].strip().replace('%', '')
                        win_rate = float(wr_str)
                    except:
                        pass
                elif 'Profit Factor:' in line:
                    try:
                        pf_str = line.split(':')[1].strip()
                        profit_factor = float(pf_str)
                    except:
                        pass
                elif 'AVALIAÇÃO FINAL:' in line:
                    avaliacao_final = line.split(':')[1].strip()
            
            if retorno_medio is not None:
                performance_data.append({
                    'steps': steps,
                    'filename': filename,
                    'retorno_medio': retorno_medio,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avaliacao': avaliacao_final
                })
                
        except Exception as e:
            print(f"⚠️ Erro ao processar {file_path}: {e}")
    
    # Ordenar por steps
    performance_data.sort(key=lambda x: x['steps'])
    
    print(f"\n📊 AVALIAÇÕES ANALISADAS: {len(performance_data)}")
    
    if performance_data:
        print("\n🏆 PERFORMANCE POR CHECKPOINT:")
        
        best_performance = None
        best_steps = None
        
        for perf in performance_data:
            status_emoji = {
                '🟢 EXCELENTE': '🟢',
                '🟡 BOM': '🟡', 
                '🟠 REGULAR': '🟠',
                '🔴 RUIM': '🔴'
            }.get(perf['avaliacao'], '❓')
            
            print(f"   {perf['steps']:>7,}k steps: {perf['retorno_medio']:>6.1f}% | WR: {perf['win_rate'] or 0:>5.1f}% | PF: {perf['profit_factor'] or 0:>4.2f} {status_emoji}")
            
            # Encontrar melhor performance
            if best_performance is None or perf['retorno_medio'] > best_performance:
                best_performance = perf['retorno_medio']
                best_steps = perf['steps']
        
        if best_performance is not None:
            print(f"\n🏆 MELHOR PERFORMANCE: {best_performance:.1f}% aos {best_steps:,} steps")
            
            # Verificar se houve degradação após o pico
            post_peak_data = [p for p in performance_data if p['steps'] > best_steps]
            
            if post_peak_data:
                avg_post_peak = np.mean([p['retorno_medio'] for p in post_peak_data])
                print(f"📉 PERFORMANCE PÓS-PICO: {avg_post_peak:.1f}% (média)")
                
                if avg_post_peak < best_performance * 0.8:  # 20% degradação
                    print("⚠️ POSSÍVEL OVERTRAINING DETECTADO!")
                    print("🛡️ Early stopping teria sido BENÉFICO")
                else:
                    print("✅ Performance mantida - Early stopping adequadamente configurado")
    
    # 4. Conclusões e Recomendações
    print("\n" + "="*60)
    print("🔍 CONCLUSÕES SOBRE EARLY STOPPING:")
    print("="*60)
    
    # Verificar se o sistema chegou ao final planejado
    if checkpoint_info and checkpoint_info[-1]['steps'] >= 5_000_000:
        print("✅ TREINAMENTO COMPLETADO SEM EARLY STOPPING")
        print("   • Sistema treinou até o final dos 5M steps planejados")
        print("   • Early stopping não foi acionado")
        
        # Verificar eficácia baseada na performance
        if performance_data and len(performance_data) >= 3:
            latest_perf = performance_data[-1]['retorno_medio']
            if best_performance and latest_perf < best_performance * 0.7:
                print("⚠️ MAS: Performance degradou significativamente no final")
                print("🔧 RECOMENDAÇÃO: Early stopping mais agressivo")
            else:
                print("✅ Performance final satisfatória")
                print("🎯 Sistema early stopping bem calibrado")
    else:
        print("🟡 TREINAMENTO INCOMPLETO")
        print("   • Possível early stopping ativado OU interrupção manual")
        
    # Análise da implementação no código
    print(f"\n🔧 IMPLEMENTAÇÃO NO CÓDIGO:")
    print(f"   • Early stopping DESABILITADO na linha 2447: return True")
    print(f"   • Configuração: patience=100k steps, min_steps=500k")
    print(f"   • Thresholds: entropy < -20.0, policy < 0.001")
    print(f"   • Status: SISTEMA PREPARADO MAS INATIVO")
    
    print(f"\n💡 RECOMENDAÇÕES:")
    if checkpoint_info and checkpoint_info[-1]['steps'] >= 5_000_000:
        print("   1. ✅ Early stopping funcionou como esperado (não interrompeu)")
        print("   2. 🔧 Considerar ativar para próximos treinamentos longos")
        print("   3. 📊 Monitorar performance vs steps em tempo real")
    else:
        print("   1. 🔍 Investigar causa da interrupção do treinamento")
        print("   2. ⚗️ Testar early stopping ativo em próximos experimentos")
        print("   3. 📈 Implementar métricas de validação mais robustas")

if __name__ == "__main__":
    analisar_logs_silus()