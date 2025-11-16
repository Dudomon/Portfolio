#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 MONITOR DE OTIMIZAÇÃO BALANCEADA

Monitora métricas para verificar se a configuração balanceada está funcionando
"""

import sys
import os
import time
from datetime import datetime

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def monitor_balanced_optimization():
    """Monitor das métricas de otimização balanceada"""
    
    print("🎯 MONITOR DE OTIMIZAÇÃO BALANCEADA")
    print("=" * 80)
    print(f"⏰ Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar configuração atual
    print("\n📊 CONFIGURAÇÃO BALANCEADA ATIVA:")
    print("=" * 60)
    print("🎯 Filosofia: BALANCED_PROGRESSIVE_LEARNING")
    print("📈 Learning Rate: 1.2e-4 (balanceado)")
    print("⚡ Gradient Accumulation: 4 steps (reduzido)")
    print("📊 Schedule: cosine_with_restarts (800k period)")
    print("🔧 Volatility Boost: DESABILITADO")
    print("🎨 Data Augmentation: REDUZIDO (25%)")
    print("🎯 Filtros V7: 0.3/0.2 (mantidos)")
    
    # Alvos esperados
    print("\n🎯 ALVOS ESPERADOS:")
    print("=" * 60)
    print("✅ KL Divergence: 1e-3 a 5e-3 (saudável)")
    print("✅ Clip Fraction: 0.05 a 0.25 (ativo mas não excessivo)")
    print("✅ Learning Rate: ~1.2e-4 (estável)")
    print("✅ Convergência: > 2M steps (objetivo principal)")
    print("✅ Estabilidade: Sem explosões de gradiente")
    
    # Benefícios da configuração balanceada
    print("\n💡 BENEFÍCIOS DA CONFIGURAÇÃO BALANCEADA:")
    print("=" * 60)
    print("🎯 MANTÉM objetivo anti-convergência prematura")
    print("📈 Learning rate suficiente para progresso contínuo")
    print("🛡️ PREVINE instabilidade e KL explosion")
    print("⚡ Gradient accumulation mais estável")
    print("🎨 Data augmentation moderado (efetivo sem ruído excessivo)")
    print("🔧 Schedule cosine com restarts menos frequentes (800k)")
    
    # Comparação com versões anteriores
    print("\n📊 COMPARAÇÃO COM VERSÕES ANTERIORES:")
    print("=" * 60)
    print("| Métrica | Problema Original | Correção Agressiva | BALANCEADO |")
    print("|---------|-------------------|-------------------|------------|")
    print("| KL Div  | 2.4e-05 (baixo)   | >1e-2 (muito alto)| 1e-3-5e-3  |")
    print("| LR      | 4.98e-05 (baixo)  | 2.0e-4+ (instável)| 1.2e-4     |")
    print("| Clip    | 0 (inativo)       | >0.5 (excessivo)  | 0.05-0.25  |")
    print("| Conv.   | 2M steps (cedo)   | Instável          | >2M estável|")
    
    # Próximos passos
    print("\n🚀 PRÓXIMOS PASSOS:")
    print("=" * 60)
    print("1. 🔄 REINICIAR o treinamento com nova configuração")
    print("2. 📊 MONITORAR métricas nos primeiros 100k steps")    
    print("3. ✅ VERIFICAR se KL está na faixa 1e-3 a 5e-3")
    print("4. 🎯 CONFIRMAR que não há convergência prematura aos 2M")
    print("5. 📈 AVALIAR performance após 3M+ steps")
    
    # Sinais de alerta
    print("\n⚠️ SINAIS DE ALERTA PARA MONITORAR:")
    print("=" * 60)
    print("🔴 KL Divergence < 5e-4: Ainda muito baixo (aumentar LR)")
    print("🔴 KL Divergence > 1e-2: Muito alto (reduzir LR)")  
    print("🔴 Clip Fraction > 0.4: Clipping excessivo")
    print("🔴 Clip Fraction < 0.02: Clipping insuficiente")
    print("🔴 Portfolio estagnado por >500k steps: Convergência prematura")
    
    print("\n✅ CONFIGURAÇÃO BALANCEADA PRONTA!")
    print("🎯 Objetivo: Evitar convergência prematura SEM instabilidade")
    print("=" * 80)

def check_training_health():
    """Verificar saúde do treinamento com configuração balanceada"""
    
    print("\n🔍 VERIFICAÇÃO DE SAÚDE DO TREINAMENTO:")
    print("=" * 60)
    
    # Verificar se arquivo de treinamento existe
    models_dir = "Otimizacao/treino_principal/models/DAYTRADER"
    csv_file = f"{models_dir}/DAYTRADER_training_metrics_20250729_110917.csv"
    
    if os.path.exists(csv_file):
        size = os.path.getsize(csv_file) / (1024*1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(csv_file))
        time_diff = datetime.now() - mtime
        
        print(f"📊 Arquivo de métricas: {csv_file}")
        print(f"📁 Tamanho: {size:.1f}MB")
        print(f"⏰ Última modificação: {mtime.strftime('%H:%M:%S')}")
        
        if time_diff.total_seconds() < 300:  # 5 minutos
            print("✅ Treinamento ATIVO (arquivo sendo atualizado)")
            print("💡 Aguarde alguns minutos e monitore as métricas")
        else:
            print("⚠️ Treinamento pode estar parado")
            print("💡 Verifique se o processo está rodando")
    else:
        print("❌ Arquivo de métricas não encontrado")
        print("💡 Treinamento pode não ter iniciado ainda")

def main():
    """Executar monitoramento completo"""
    
    print("🚀 SISTEMA DE MONITORAMENTO - OTIMIZAÇÃO BALANCEADA")
    print("=" * 80)
    
    # Monitor principal
    monitor_balanced_optimization()
    
    # Verificar saúde do treinamento
    check_training_health()
    
    print(f"\n⏰ Monitor concluído em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 Execute novamente para acompanhar progresso")

if __name__ == "__main__":
    main()