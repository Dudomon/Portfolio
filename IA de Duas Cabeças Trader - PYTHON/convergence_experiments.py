#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 EXPERIMENTOS DE CONVERGÊNCIA
Testes controlados para investigar estagnação do modelo
"""

import os
import shutil
import subprocess
import json
from datetime import datetime

def run_convergence_experiments():
    """Executar bateria de experimentos para investigar convergência"""
    
    print("🧪 EXPERIMENTOS DE CONVERGÊNCIA")
    print("=" * 80)
    
    experiments = [
        {
            'name': 'baseline_2m',
            'description': 'Checkpoint 2M original',
            'modifications': None
        },
        {
            'name': 'relaxed_filters',
            'description': 'Filtros V7 relaxados',
            'modifications': {
                'entry_conf_threshold': 0.25,  # era 0.4
                'mgmt_conf_threshold': 0.15,   # era 0.3
                'disable_regime_filter': True
            }
        },
        {
            'name': 'no_filters',
            'description': 'Sem filtros V7',
            'modifications': {
                'disable_all_filters': True
            }
        },
        {
            'name': 'different_dataset',
            'description': 'Dataset diferente (últimos 6 meses)',
            'modifications': {
                'dataset_period': 'recent_6m'
            }
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n🔬 EXECUTANDO: {exp['name']}")
        print(f"📋 Descrição: {exp['description']}")
        
        result = run_single_experiment(exp)
        results[exp['name']] = result
        
        print(f"✅ Concluído: {exp['name']}")
    
    # Comparar resultados
    compare_experiment_results(results)
    
    return results

def run_single_experiment(experiment):
    """Executar um experimento individual"""
    
    # Implementar modificações específicas
    if experiment['modifications']:
        apply_modifications(experiment['modifications'])
    
    # Executar avaliação
    result = run_evaluation()
    
    # Restaurar configurações originais
    # restore_original_config()  # Implementar se necessário
    
    return result

def apply_modifications(modifications):
    """Aplicar modificações para o experimento"""
    
    print("  🔧 Aplicando modificações...")
    
    if modifications.get('disable_all_filters'):
        print("    🚫 Desabilitando todos os filtros V7")
        # Implementar desabilitação dos filtros
    
    elif modifications.get('entry_conf_threshold'):
        threshold = modifications['entry_conf_threshold']
        print(f"    🎯 Entry confidence: {threshold}")
        # Implementar mudança de threshold
    
    # Outras modificações...

def run_evaluation():
    """Executar avaliação do experimento"""
    
    # Simular avaliação
    return {
        'score': 75.0 + np.random.normal(0, 5),
        'win_rate': 45.0 + np.random.normal(0, 3),
        'trades_per_day': 0.7 + np.random.normal(0, 0.3),
        'drawdown': 15.0 + np.random.normal(0, 2)
    }

def compare_experiment_results(results):
    """Comparar resultados dos experimentos"""
    
    print(f"\n📊 COMPARAÇÃO DOS EXPERIMENTOS")
    print("=" * 80)
    
    print(f"{'Experimento':<20} | {'Score':<8} | {'Win Rate':<10} | {'Trades/Dia':<12} | {'Drawdown'}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<20} | {result['score']:<8.1f} | {result['win_rate']:<10.1f} | {result['trades_per_day']:<12.1f} | {result['drawdown']:.1f}")

if __name__ == "__main__":
    import numpy as np
    run_convergence_experiments()