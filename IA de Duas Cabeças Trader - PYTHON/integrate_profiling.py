#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 INTEGRAÇÃO DO PROFILING NO DAYTRADER
Script para adicionar profiling em tempo real ao daytrader.py
"""

import os
import re

def integrate_profiling_into_daytrader():
    """Integrar profiling no daytrader.py"""
    
    daytrader_file = "daytrader.py"
    
    if not os.path.exists(daytrader_file):
        print(f"❌ Arquivo {daytrader_file} não encontrado!")
        return
    
    # Ler arquivo original
    with open(daytrader_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar se já foi integrado
    if "from dayprofile import" in content:
        print("✅ Profiling já integrado no daytrader.py")
        return
    
    # Adicionar imports do profiling
    profiling_imports = """
# 🔍 SISTEMA DE PROFILING EM TEMPO REAL
from dayprofile import start_profiling, stop_profiling, ProfileContext, profile_training_step, get_profiling_stats
"""
    
    # Encontrar local para inserir imports (após outros imports)
    import_pattern = r"(from trading_framework\..*?import.*?\n)"
    matches = list(re.finditer(import_pattern, content))
    
    if matches:
        # Inserir após o último import do trading_framework
        last_match = matches[-1]
        insert_pos = last_match.end()
        content = content[:insert_pos] + profiling_imports + content[insert_pos:]
    else:
        # Inserir no início se não encontrar padrão
        content = profiling_imports + content
    
    # Adicionar profiling na função principal de treinamento
    # Procurar pela função main ou similar
    main_patterns = [
        r"(def main\(\):.*?\n)",
        r"(if __name__ == ['\"]__main__['\"]:\s*\n)",
        r"(def train_model\(.*?\):.*?\n)"
    ]
    
    for pattern in main_patterns:
        if re.search(pattern, content):
            # Adicionar start_profiling no início
            content = re.sub(
                pattern,
                r"\1    # 🔍 Iniciar profiling em tempo real\n    start_profiling()\n    \n",
                content,
                count=1
            )
            break
    
    # Adicionar ProfileContext em funções críticas
    critical_functions = [
        "model.learn",
        "env.step",
        "model.predict",
        "calculate_reward",
        "process_observation"
    ]
    
    for func in critical_functions:
        # Procurar chamadas da função e envolver com ProfileContext
        pattern = rf"(\s+)({func}\([^)]*\))"
        replacement = rf"\1with ProfileContext('{func}'):\n\1    result = \2\n\1result"
        content = re.sub(pattern, replacement, content)
    
    # Adicionar stop_profiling no final
    if "if __name__ == '__main__':" in content:
        # Adicionar try/finally para garantir que profiling pare
        content = content.replace(
            "if __name__ == '__main__':",
            """if __name__ == '__main__':
    try:"""
        )
        
        # Adicionar finally no final do arquivo
        content += """
    except KeyboardInterrupt:
        print("\\n⚠️ Treinamento interrompido pelo usuário")
    finally:
        # 🛑 Parar profiling e gerar relatório
        stop_profiling()
        print("📋 Relatório de profiling gerado em profiling_results/")
"""
    
    # Salvar arquivo modificado
    backup_file = "daytrader_backup.py"
    
    # Fazer backup
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Profiling integrado no daytrader.py")
    print(f"📋 Backup salvo em: {backup_file}")
    print(f"🔍 Para usar: execute o daytrader.py normalmente")
    print(f"📊 Relatórios serão salvos em: profiling_results/")

def create_profiling_config():
    """Criar arquivo de configuração para profiling"""
    
    config_content = """# 🔍 CONFIGURAÇÃO DO PROFILING
# Ajuste estas configurações conforme necessário

PROFILING_CONFIG = {
    # Intervalo de coleta de métricas (segundos)
    "profile_interval": 0.1,  # 100ms - alta frequência
    
    # Intervalo de análise detalhada (segundos)
    "detailed_interval": 1.0,  # 1s - análises complexas
    
    # Tamanho do buffer de métricas
    "buffer_size": 10000,  # ~16 minutos de dados
    
    # Alertas
    "cpu_alert_threshold": 90,  # %
    "memory_alert_threshold": 85,  # %
    "gpu_memory_alert_mb": 8000,  # MB
    
    # Arquivos de output
    "output_dir": "profiling_results",
    "save_detailed_logs": True,
    "save_csv_metrics": True,
    
    # Funções para profile detalhado
    "profile_functions": [
        "model.learn",
        "env.step", 
        "model.predict",
        "calculate_reward",
        "process_observation",
        "_process_v7_action",
        "_check_entry_filters",
        "_apply_v7_intuition_filters"
    ],
    
    # GPU monitoring
    "enable_gpu_monitoring": True,
    
    # Memory leak detection
    "memory_leak_threshold_mb": 100,  # MB growth
    "memory_leak_window": 20,  # samples
}
"""
    
    with open("profiling_config.py", 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("⚙️ Arquivo de configuração criado: profiling_config.py")

def create_profiling_analysis_script():
    """Criar script para análise dos resultados de profiling"""
    
    analysis_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
📊 ANÁLISE DOS RESULTADOS DE PROFILING
Script para analisar e visualizar os dados coletados pelo profiling
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from datetime import datetime

def analyze_profiling_results(results_dir="profiling_results"):
    '''Analisar resultados de profiling'''
    
    # Encontrar arquivo CSV mais recente
    csv_files = glob.glob(f"{results_dir}/realtime_metrics_*.csv")
    
    if not csv_files:
        print("❌ Nenhum arquivo de profiling encontrado!")
        return
    
    latest_csv = sorted(csv_files)[-1]
    print(f"📊 Analisando: {latest_csv}")
    
    # Carregar dados
    df = pd.read_csv(latest_csv)
    
    # Análise básica
    print("\\n📈 ESTATÍSTICAS GERAIS:")
    print(f"Duração total: {df['elapsed_time'].max():.1f} segundos")
    print(f"Amostras coletadas: {len(df):,}")
    print(f"CPU médio: {df['cpu_percent'].mean():.1f}%")
    print(f"Memória média: {df['memory_mb'].mean():.1f}MB")
    print(f"GPU médio: {df['gpu_utilization'].mean():.1f}%")
    
    # Detectar gargalos
    print("\\n🔍 GARGALOS DETECTADOS:")
    
    cpu_high = df[df['cpu_percent'] > 85]
    if len(cpu_high) > 0:
        print(f"⚠️ CPU alto: {len(cpu_high)} amostras ({len(cpu_high)/len(df)*100:.1f}%)")
    
    memory_high = df[df['memory_percent'] > 80]
    if len(memory_high) > 0:
        print(f"⚠️ Memória alta: {len(memory_high)} amostras ({len(memory_high)/len(df)*100:.1f}%)")
    
    gpu_high = df[df['gpu_utilization'] > 95]
    if len(gpu_high) > 0:
        print(f"⚠️ GPU saturada: {len(gpu_high)} amostras ({len(gpu_high)/len(df)*100:.1f}%)")
    
    # Análise de tendências
    print("\\n📈 TENDÊNCIAS:")
    
    # CPU trend
    cpu_trend = np.corrcoef(df.index, df['cpu_percent'])[0, 1]
    print(f"CPU trend: {cpu_trend:.3f} ({'crescente' if cpu_trend > 0.1 else 'decrescente' if cpu_trend < -0.1 else 'estável'})")
    
    # Memory trend
    memory_trend = np.corrcoef(df.index, df['memory_mb'])[0, 1]
    print(f"Memory trend: {memory_trend:.3f} ({'crescente' if memory_trend > 0.1 else 'decrescente' if memory_trend < -0.1 else 'estável'})")
    
    # Vazamento de memória
    memory_growth = df['memory_mb'].iloc[-1] - df['memory_mb'].iloc[0]
    print(f"Crescimento de memória: {memory_growth:.1f}MB")
    
    if memory_growth > 200:
        print("⚠️ POSSÍVEL VAZAMENTO DE MEMÓRIA!")
    
    # Gerar gráficos
    generate_profiling_charts(df, results_dir)
    
    return df

def generate_profiling_charts(df, output_dir):
    '''Gerar gráficos de análise'''
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📊 Análise de Profiling em Tempo Real', fontsize=16)
    
    # CPU Usage
    axes[0, 0].plot(df['elapsed_time'], df['cpu_percent'], color='#ff6b6b', alpha=0.7)
    axes[0, 0].set_title('🖥️ CPU Usage (%)')
    axes[0, 0].set_xlabel('Tempo (s)')
    axes[0, 0].set_ylabel('CPU %')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Memory Usage
    axes[0, 1].plot(df['elapsed_time'], df['memory_mb'], color='#4ecdc4', alpha=0.7)
    axes[0, 1].set_title('💾 Memory Usage (MB)')
    axes[0, 1].set_xlabel('Tempo (s)')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # GPU Usage
    axes[1, 0].plot(df['elapsed_time'], df['gpu_utilization'], color='#45b7d1', alpha=0.7)
    axes[1, 0].set_title('🎮 GPU Usage (%)')
    axes[1, 0].set_xlabel('Tempo (s)')
    axes[1, 0].set_ylabel('GPU %')
    axes[1, 0].grid(True, alpha=0.3)
    
    # PyTorch Memory
    axes[1, 1].plot(df['elapsed_time'], df['torch_allocated_mb'], color='#f7b731', alpha=0.7, label='Allocated')
    axes[1, 1].plot(df['elapsed_time'], df['torch_cached_mb'], color='#5f27cd', alpha=0.7, label='Cached')
    axes[1, 1].set_title('🔥 PyTorch Memory (MB)')
    axes[1, 1].set_xlabel('Tempo (s)')
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    chart_file = f"{output_dir}/profiling_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"📊 Gráficos salvos em: {chart_file}")
    
    plt.show()

def generate_recommendations(df):
    '''Gerar recomendações baseadas na análise'''
    
    recommendations = []
    
    # CPU
    avg_cpu = df['cpu_percent'].mean()
    if avg_cpu > 80:
        recommendations.append("🖥️ CPU usage alto - considere otimizar algoritmos ou reduzir batch size")
    elif avg_cpu < 30:
        recommendations.append("🖥️ CPU subutilizado - considere aumentar batch size ou paralelização")
    
    # Memory
    memory_growth = df['memory_mb'].iloc[-1] - df['memory_mb'].iloc[0]
    if memory_growth > 200:
        recommendations.append("💾 Possível vazamento de memória - verifique limpeza de variáveis")
    
    # GPU
    avg_gpu = df['gpu_utilization'].mean()
    if avg_gpu < 50:
        recommendations.append("🎮 GPU subutilizada - considere aumentar batch size ou usar mixed precision")
    elif avg_gpu > 95:
        recommendations.append("🎮 GPU saturada - considere reduzir batch size")
    
    # PyTorch Memory
    max_torch_alloc = df['torch_allocated_mb'].max()
    if max_torch_alloc > 8000:  # 8GB
        recommendations.append("🔥 Alto uso de VRAM - considere gradient checkpointing ou batch size menor")
    
    print("\\n💡 RECOMENDAÇÕES:")
    for rec in recommendations:
        print(f"   {rec}")
    
    if not recommendations:
        print("   ✅ Sistema operando dentro dos parâmetros normais")

if __name__ == "__main__":
    print("📊 ANÁLISE DE PROFILING")
    print("=" * 60)
    
    df = analyze_profiling_results()
    
    if df is not None:
        generate_recommendations(df)
        
        print("\\n✅ Análise concluída!")
        print("📊 Verifique os gráficos gerados")
"""
    
    with open("analyze_profiling.py", 'w', encoding='utf-8') as f:
        f.write(analysis_script)
    
    print("📊 Script de análise criado: analyze_profiling.py")

def main():
    """Função principal"""
    print("🔧 INTEGRAÇÃO DO PROFILING NO DAYTRADER")
    print("=" * 60)
    
    print("1. Integrando profiling no daytrader.py...")
    integrate_profiling_into_daytrader()
    
    print("\\n2. Criando configuração de profiling...")
    create_profiling_config()
    
    print("\\n3. Criando script de análise...")
    create_profiling_analysis_script()
    
    print("\\n✅ INTEGRAÇÃO COMPLETA!")
    print("=" * 60)
    print("📋 PRÓXIMOS PASSOS:")
    print("1. Execute: python daytrader.py")
    print("2. O profiling iniciará automaticamente")
    print("3. Após o treinamento, execute: python analyze_profiling.py")
    print("4. Verifique os relatórios em: profiling_results/")
    print("=" * 60)

if __name__ == "__main__":
    main()