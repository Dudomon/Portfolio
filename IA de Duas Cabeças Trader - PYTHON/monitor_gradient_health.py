#!/usr/bin/env python3
"""
🔍 MONITOR GRADIENT HEALTH - Monitorar se os zeros extremos diminuíram
"""

import time
import re
import os
from datetime import datetime

def monitor_gradient_health():
    """Monitorar saúde dos gradientes em tempo real"""
    
    print("🔍 MONITOR GRADIENT HEALTH - Monitorando zeros extremos")
    print("=" * 60)
    print("🎯 OBJETIVO: Zeros extremos < 30% (era 54-66%)")
    print("🔧 MUDANÇAS: LR 2.67e-05 → 3e-04, max_grad_norm 0.3 → 1.0")
    print("=" * 60)
    
    last_check_time = 0
    zero_history = []
    
    while True:
        try:
            # Procurar por arquivos de debug recentes
            debug_files = []
            for file in os.listdir('.'):
                if file.startswith('debug_zeros_report_step_') and file.endswith('.txt'):
                    debug_files.append(file)
            
            if not debug_files:
                print("⏳ Aguardando arquivos de debug...")
                time.sleep(10)
                continue
            
            # Pegar o arquivo mais recente
            latest_file = max(debug_files, key=os.path.getmtime)
            file_time = os.path.getmtime(latest_file)
            
            # Só processar se for novo
            if file_time <= last_check_time:
                time.sleep(5)
                continue
            
            last_check_time = file_time
            
            # Ler e analisar o arquivo
            with open(latest_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extrair step number
            step_match = re.search(r'step_(\d+)', latest_file)
            step = step_match.group(1) if step_match else "unknown"
            
            # Procurar por zeros extremos
            zero_matches = re.findall(r'(\d+\.\d+)% zeros extremos', content)
            
            if zero_matches:
                zero_percentages = [float(match) for match in zero_matches]
                avg_zeros = sum(zero_percentages) / len(zero_percentages)
                max_zeros = max(zero_percentages)
                min_zeros = min(zero_percentages)
                
                zero_history.append({
                    'step': step,
                    'avg_zeros': avg_zeros,
                    'max_zeros': max_zeros,
                    'min_zeros': min_zeros,
                    'count': len(zero_percentages)
                })
                
                # Manter só últimos 10 registros
                if len(zero_history) > 10:
                    zero_history = zero_history[-10:]
                
                # Status atual
                status = "🟢 BOM" if avg_zeros < 30 else "🟡 MODERADO" if avg_zeros < 50 else "🔴 CRÍTICO"
                
                print(f"\n📊 STEP {step} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"   Zeros Médios: {avg_zeros:.1f}% {status}")
                print(f"   Range: {min_zeros:.1f}% - {max_zeros:.1f}%")
                print(f"   Camadas: {len(zero_percentages)}")
                
                # Tendência
                if len(zero_history) >= 3:
                    recent_avg = sum(h['avg_zeros'] for h in zero_history[-3:]) / 3
                    older_avg = sum(h['avg_zeros'] for h in zero_history[-6:-3]) / 3 if len(zero_history) >= 6 else recent_avg
                    
                    if recent_avg < older_avg - 5:
                        trend = "📈 MELHORANDO"
                    elif recent_avg > older_avg + 5:
                        trend = "📉 PIORANDO"
                    else:
                        trend = "➡️ ESTÁVEL"
                    
                    print(f"   Tendência: {trend}")
                
                # Alerta se ainda crítico
                if avg_zeros > 50:
                    print(f"   ⚠️ AINDA CRÍTICO: {avg_zeros:.1f}% > 50%")
                    print(f"   💡 SUGESTÃO: Considerar LR ainda maior ou inicialização diferente")
                elif avg_zeros < 30:
                    print(f"   ✅ OBJETIVO ATINGIDO: {avg_zeros:.1f}% < 30%")
            
            time.sleep(10)  # Check a cada 10 segundos
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoramento interrompido")
            break
        except Exception as e:
            print(f"❌ Erro no monitoramento: {e}")
            time.sleep(5)
    
    # Resumo final
    if zero_history:
        print(f"\n📋 RESUMO FINAL:")
        print(f"   Registros: {len(zero_history)}")
        final_avg = zero_history[-1]['avg_zeros']
        initial_avg = zero_history[0]['avg_zeros']
        improvement = initial_avg - final_avg
        
        print(f"   Inicial: {initial_avg:.1f}%")
        print(f"   Final: {final_avg:.1f}%")
        print(f"   Melhoria: {improvement:+.1f}%")
        
        if improvement > 10:
            print(f"   🎉 GRANDE MELHORIA!")
        elif improvement > 0:
            print(f"   ✅ MELHORIA DETECTADA")
        else:
            print(f"   ⚠️ SEM MELHORIA SIGNIFICATIVA")

if __name__ == "__main__":
    monitor_gradient_health()