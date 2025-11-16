#!/usr/bin/env python3
"""
🎯 MONITOR OVERTRAINING V7 - Detector especializado para arquitetura TwoHeadV7Intuition
Sistema que detecta overtraining em modelos não-determinísticos através de múltiplas runs
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
from datetime import datetime
import time
import torch
from collections import defaultdict
import json

# ========== CONFIGURAÇÃO ==========
CHECKPOINT_NAME = "DAYTRADER_phase1fundamentalsextended_2550000_steps_20250815_093921.zip"  # 🎯 DAYTRADER 2.55M atual
N_RUNS = 10  # Múltiplas runs não-determinísticas
SAMPLES_PER_RUN = 1000  # Samples por run
OVERTRAINING_THRESHOLDS = {
    'quality_variance': 0.3,      # Variância Entry Quality > 0.3 = suspeito
    'performance_cv': 1.5,        # CV performance > 1.5 = instável  
    'zero_percentage': 0.7,       # >70% zeros = problema
    'extreme_concentration': 0.8  # >80% em extremos (0 ou 1) = overtraining
}
# ==================================

def run_single_evaluation(model, run_id):
    """🧪 Executar uma avaliação individual não-determinística"""
    
    # Estados LSTM
    lstm_states = None
    results = {
        'run_id': run_id,
        'entry_qualities': [],
        'entry_decisions': [],
        'actions': []
    }
    
    print(f"  🔄 Run {run_id+1}/{N_RUNS}...")
    
    for i in range(SAMPLES_PER_RUN):
        # Observação sintética (normalizada)
        obs = np.random.normal(0, 0.5, (2580,)).astype(np.float32)
        
        # Predição NÃO-DETERMINÍSTICA
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states,
            deterministic=False  # 🔥 CHAVE: modo não-determinístico
        )
        
        entry_decision = int(action[0])
        entry_quality = float(action[1])
        
        results['entry_qualities'].append(entry_quality)
        results['entry_decisions'].append(entry_decision)
        results['actions'].append(action.copy())
    
    return results

def analyze_overtraining_patterns(all_results):
    """📊 Analisar padrões de overtraining across múltiplas runs"""
    
    print("\n🔬 ANÁLISE DE OVERTRAINING PATTERNS")
    print("=" * 60)
    
    # Consolidar dados de todas as runs
    all_qualities = []
    run_stats = []
    
    for result in all_results:
        qualities = np.array(result['entry_qualities'])
        decisions = np.array(result['entry_decisions'])
        
        # Stats por run
        run_stat = {
            'mean_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'zero_percentage': np.sum(qualities == 0.0) / len(qualities),
            'one_percentage': np.sum(qualities == 1.0) / len(qualities),
            'trades_attempted': np.sum(decisions > 0),
            'hold_percentage': np.sum(decisions == 0) / len(decisions)
        }
        run_stats.append(run_stat)
        all_qualities.extend(qualities)
    
    all_qualities = np.array(all_qualities)
    
    # Métricas de overtraining
    overtraining_metrics = {}
    
    # 1. VARIÂNCIA ENTRE RUNS (consistência)
    means_across_runs = [stat['mean_quality'] for stat in run_stats]
    stds_across_runs = [stat['std_quality'] for stat in run_stats]
    
    overtraining_metrics['inter_run_variance'] = np.var(means_across_runs)
    overtraining_metrics['mean_std_variance'] = np.var(stds_across_runs)
    
    # 2. COEFFICIENT OF VARIATION (estabilidade)
    mean_performance = np.mean(means_across_runs)
    std_performance = np.std(means_across_runs)
    overtraining_metrics['performance_cv'] = std_performance / mean_performance if mean_performance > 0 else float('inf')
    
    # 3. CONCENTRAÇÃO EM EXTREMOS (0 e 1)
    zero_perc = np.mean([stat['zero_percentage'] for stat in run_stats])
    one_perc = np.mean([stat['one_percentage'] for stat in run_stats])
    overtraining_metrics['zero_percentage'] = zero_perc
    overtraining_metrics['one_percentage'] = one_perc
    overtraining_metrics['extreme_concentration'] = zero_perc + one_perc
    
    # 4. BIMODALIDADE (sinal de instabilidade)
    hist, bins = np.histogram(all_qualities, bins=20, range=(0, 1))
    # Detectar picos múltiplos
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1:
            peaks.append(i)
    
    overtraining_metrics['n_peaks'] = len(peaks)
    overtraining_metrics['bimodal_strength'] = len(peaks) >= 2
    
    # 5. RANGE CONSISTENCY (todas as runs deveriam ter ranges similares)
    ranges = [stat['max_quality'] - stat['min_quality'] for stat in run_stats]
    overtraining_metrics['range_consistency'] = np.std(ranges)
    
    return overtraining_metrics, run_stats, all_qualities

def evaluate_overtraining_severity(metrics):
    """🚨 Avaliar severidade do overtraining"""
    
    warnings = []
    critical_issues = []
    
    # Checks de overtraining
    if metrics['performance_cv'] > OVERTRAINING_THRESHOLDS['performance_cv']:
        critical_issues.append(f"Performance CV muito alta: {metrics['performance_cv']:.3f} > {OVERTRAINING_THRESHOLDS['performance_cv']}")
    
    if metrics['zero_percentage'] > OVERTRAINING_THRESHOLDS['zero_percentage']:
        critical_issues.append(f"Muitos zeros: {metrics['zero_percentage']*100:.1f}% > {OVERTRAINING_THRESHOLDS['zero_percentage']*100}%")
    
    if metrics['extreme_concentration'] > OVERTRAINING_THRESHOLDS['extreme_concentration']:
        warnings.append(f"Concentração em extremos: {metrics['extreme_concentration']*100:.1f}% > {OVERTRAINING_THRESHOLDS['extreme_concentration']*100}%")
    
    if metrics['inter_run_variance'] > OVERTRAINING_THRESHOLDS['quality_variance']:
        warnings.append(f"Alta variância entre runs: {metrics['inter_run_variance']:.3f} > {OVERTRAINING_THRESHOLDS['quality_variance']}")
    
    if metrics['bimodal_strength']:
        warnings.append(f"Distribuição bimodal detectada ({metrics['n_peaks']} picos)")
    
    if metrics['range_consistency'] > 0.3:
        warnings.append(f"Ranges inconsistentes entre runs: {metrics['range_consistency']:.3f}")
    
    # Determinar severidade
    if len(critical_issues) >= 2:
        severity = "🔥 OVERTRAINING SEVERO"
        color = "critical"
    elif len(critical_issues) >= 1:
        severity = "⚠️ OVERTRAINING MODERADO"
        color = "warning"
    elif len(warnings) >= 3:
        severity = "🔶 SUSPEITA DE OVERTRAINING"
        color = "warning"
    elif len(warnings) >= 1:
        severity = "✅ MODELO ESTÁVEL (alguns avisos)"
        color = "ok"
    else:
        severity = "✅ MODELO SAUDÁVEL"
        color = "ok"
    
    return severity, critical_issues, warnings, color

def main():
    # Extrair steps do checkpoint
    import re
    steps_match = re.search(r'(\d+)_steps', CHECKPOINT_NAME)
    steps = int(steps_match.group(1)) if steps_match else 0
    steps_millions = f"{steps/1000000:.1f}M"
    
    print(f"🎯 MONITOR OVERTRAINING V7 - CHECKPOINT {steps_millions}")
    print("=" * 70)
    print(f"🔥 Arquitetura: TwoHeadV7Intuition (modo não-determinístico)")
    print(f"📊 Runs: {N_RUNS} × {SAMPLES_PER_RUN} samples = {N_RUNS * SAMPLES_PER_RUN:,} total")
    print("=" * 70)
    
    try:
        # Imports
        from sb3_contrib import RecurrentPPO
        print("✅ Imports ok")
        
        # Carregar modelo - mesmo caminho da avaliação anterior
        checkpoint_path = f"D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/{CHECKPOINT_NAME}"
        print(f"🤖 Carregando modelo {steps_millions}...")
        model = RecurrentPPO.load(checkpoint_path, device='cuda')
        model.policy.set_training_mode(True)  # 🔥 MODO STOCHASTIC para detecção correta
        print(f"✅ Modelo carregado em modo stochastic")
        
        # Executar múltiplas runs
        print(f"\n🔄 EXECUTANDO {N_RUNS} RUNS NÃO-DETERMINÍSTICAS...")
        all_results = []
        
        start_time = time.time()
        for run_id in range(N_RUNS):
            result = run_single_evaluation(model, run_id)
            all_results.append(result)
        
        total_time = time.time() - start_time
        print(f"✅ Concluído em {total_time:.1f}s")
        
        # Análise de overtraining
        metrics, run_stats, all_qualities = analyze_overtraining_patterns(all_results)
        
        # Avaliação de severidade
        severity, critical_issues, warnings, color = evaluate_overtraining_severity(metrics)
        
        # RELATÓRIO DETALHADO
        print("\n" + "=" * 70)
        print("📊 RELATÓRIO DETALHADO DE OVERTRAINING")
        print("=" * 70)
        
        print(f"\n🎯 VEREDICTO FINAL: {severity}")
        
        print(f"\n📈 MÉTRICAS CRÍTICAS:")
        print(f"  Performance CV: {metrics['performance_cv']:.3f} (threshold: {OVERTRAINING_THRESHOLDS['performance_cv']})")
        print(f"  % Zeros: {metrics['zero_percentage']*100:.1f}% (threshold: {OVERTRAINING_THRESHOLDS['zero_percentage']*100}%)")
        print(f"  % Extremos: {metrics['extreme_concentration']*100:.1f}% (threshold: {OVERTRAINING_THRESHOLDS['extreme_concentration']*100}%)")
        print(f"  Variância inter-runs: {metrics['inter_run_variance']:.3f} (threshold: {OVERTRAINING_THRESHOLDS['quality_variance']})")
        
        print(f"\n📊 ESTATÍSTICAS DETALHADAS:")
        means = [stat['mean_quality'] for stat in run_stats]
        print(f"  Entry Quality média: {np.mean(means):.3f} ± {np.std(means):.3f}")
        print(f"  Range médio: {np.mean([stat['max_quality'] - stat['min_quality'] for stat in run_stats]):.3f}")
        print(f"  Consistência de range: {metrics['range_consistency']:.3f}")
        print(f"  Distribuição: {metrics['n_peaks']} picos {'(bimodal)' if metrics['bimodal_strength'] else '(unimodal)'}")
        
        if critical_issues:
            print(f"\n🔥 PROBLEMAS CRÍTICOS:")
            for issue in critical_issues:
                print(f"  • {issue}")
        
        if warnings:
            print(f"\n⚠️ AVISOS:")
            for warning in warnings:
                print(f"  • {warning}")
        
        # RECOMENDAÇÕES
        print(f"\n💡 RECOMENDAÇÕES:")
        if color == "critical":
            print("  🚨 PARAR TREINAMENTO IMEDIATAMENTE")
            print("  🔄 Fazer rollback para checkpoint anterior")
            print("  📉 Modelo apresenta instabilidade severa")
        elif color == "warning":
            print("  ⚠️ Monitorar de perto próximos checkpoints")
            print("  🎯 Considerar early stopping em breve")
            print("  📊 Avaliar se performance ainda melhora")
        else:
            print("  ✅ Modelo operando dentro dos parâmetros normais")
            print("  📈 Pode continuar treinamento com monitoramento")
        
        # Salvar relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"D:/Projeto/avaliacoes/overtraining_EXPERTGAIN_{steps}_steps_{timestamp}.json"
        
        report = {
            'checkpoint': CHECKPOINT_NAME,
            'steps': steps,
            'timestamp': timestamp,
            'severity': severity,
            'metrics': metrics,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'run_stats': run_stats,
            'config': {
                'n_runs': N_RUNS,
                'samples_per_run': SAMPLES_PER_RUN,
                'thresholds': OVERTRAINING_THRESHOLDS
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Relatório salvo: {filename}")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()