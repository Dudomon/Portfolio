#!/usr/bin/env python3
"""
🔍 ANALISADOR DE LOGS JSONL - Treinamento SILUS em Tempo Real
Analisa os arquivos JSONL do treinamento atual na pasta avaliacoes
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def find_jsonl_files():
    """🔍 Encontra todos os arquivos JSONL na pasta avaliacoes"""
    patterns = [
        "D:/Projeto/avaliacoes/*.jsonl",
        "D:/Projeto/avaliacoes/**/*.jsonl"
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    if not all_files:
        print("❌ Nenhum arquivo JSONL encontrado na pasta avaliacoes")
        return []
    
    # Filtrar por data de modificação (mais recentes primeiro)
    files_with_time = [(f, os.path.getmtime(f)) for f in all_files]
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    print(f"🔍 Encontrados {len(all_files)} arquivos JSONL:")
    for i, (filepath, mod_time) in enumerate(files_with_time[:10]):  # Top 10
        mod_datetime = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        filename = os.path.basename(filepath)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {i+1:2d}. {filename} ({size_kb:.1f}KB) - {mod_datetime}")
    
    return [f[0] for f in files_with_time]

def load_jsonl_data(filepath):
    """📊 Carrega dados de um arquivo JSONL"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    if line_num <= 5:  # Só reportar primeiros erros
                        print(f"⚠️ Erro JSON linha {line_num}: {str(e)[:50]}...")
    except Exception as e:
        print(f"❌ Erro ao ler {os.path.basename(filepath)}: {e}")
        return []
    
    return data

def analyze_convergence_data(data, filename):
    """📈 Analisa dados de convergência"""
    if not data:
        return
    
    print(f"\n🔍 ANÁLISE CONVERGÊNCIA: {filename}")
    print("=" * 50)
    
    # Estatísticas básicas
    total_entries = len(data)
    print(f"📊 Total de entradas: {total_entries:,}")
    
    if total_entries == 0:
        return
    
    # Extrair métricas comuns
    steps = []
    rewards = []
    losses = []
    gradients = []
    
    for entry in data:
        if 'step' in entry:
            steps.append(entry['step'])
        if 'reward' in entry:
            rewards.append(entry['reward'])
        if 'total_loss' in entry:
            losses.append(entry['total_loss'])
        if 'gradient_norm' in entry:
            gradients.append(entry['gradient_norm'])
    
    # Análise de steps
    if steps:
        print(f"🎯 Steps: {min(steps):,} → {max(steps):,} ({len(steps):,} amostras)")
        recent_steps = [s for s in steps if s >= max(steps) - 10000]  # Últimos 10k steps
        print(f"🔥 Últimos 10k steps: {len(recent_steps):,} amostras")
    
    # Análise de rewards
    if rewards:
        rewards = [r for r in rewards if isinstance(r, (int, float)) and not np.isnan(r)]
        if rewards:
            print(f"💰 Rewards: μ={np.mean(rewards):.3f}, σ={np.std(rewards):.3f}")
            print(f"    Range: [{min(rewards):.3f}, {max(rewards):.3f}]")
            
            # Trend de rewards (últimos 20%)
            recent_count = max(1, len(rewards) // 5)
            recent_rewards = rewards[-recent_count:]
            older_rewards = rewards[:recent_count] if len(rewards) > recent_count else rewards
            
            if len(recent_rewards) > 0 and len(older_rewards) > 0:
                trend_change = np.mean(recent_rewards) - np.mean(older_rewards)
                trend_emoji = "📈" if trend_change > 0 else "📉" if trend_change < 0 else "➡️"
                print(f"    Tendência: {trend_emoji} {trend_change:+.3f} (últimos 20%)")
    
    # Análise de loss
    if losses:
        losses = [l for l in losses if isinstance(l, (int, float)) and not np.isnan(l)]
        if losses:
            print(f"📉 Loss: μ={np.mean(losses):.4f}, σ={np.std(losses):.4f}")
            print(f"    Range: [{min(losses):.4f}, {max(losses):.4f}]")
    
    # Análise de gradientes
    if gradients:
        gradients = [g for g in gradients if isinstance(g, (int, float)) and not np.isnan(g)]
        if gradients:
            print(f"🎯 Gradients: μ={np.mean(gradients):.4f}, σ={np.std(gradients):.4f}")
            print(f"    Range: [{min(gradients):.4f}, {max(gradients):.4f}]")
            
            # Verificar gradientes explodindo/desaparecendo
            high_grad_count = sum(1 for g in gradients if g > 1.0)
            low_grad_count = sum(1 for g in gradients if g < 0.001)
            
            if high_grad_count > len(gradients) * 0.1:
                print(f"    ⚠️ Gradientes altos: {high_grad_count}/{len(gradients)} ({high_grad_count/len(gradients)*100:.1f}%)")
            if low_grad_count > len(gradients) * 0.1:
                print(f"    ⚠️ Gradientes baixos: {low_grad_count}/{len(gradients)} ({low_grad_count/len(gradients)*100:.1f}%)")
    
    # Mostrar algumas entradas de exemplo
    print(f"\n📋 AMOSTRA DOS DADOS (últimas 3 entradas):")
    for i, entry in enumerate(data[-3:], 1):
        print(f"  {i}. {json.dumps(entry, indent=None)[:100]}...")

def analyze_performance_data(data, filename):
    """💼 Analisa dados de performance de trading"""
    if not data:
        return
    
    print(f"\n💼 ANÁLISE PERFORMANCE: {filename}")
    print("=" * 50)
    
    total_entries = len(data)
    print(f"📊 Total de entradas: {total_entries:,}")
    
    # Extrair métricas de trading
    portfolio_values = []
    win_rates = []
    drawdowns = []
    trades_counts = []
    
    for entry in data:
        if 'portfolio_value' in entry:
            portfolio_values.append(entry['portfolio_value'])
        if 'win_rate' in entry:
            win_rates.append(entry['win_rate'])
        if 'max_drawdown' in entry:
            drawdowns.append(entry['max_drawdown'])
        if 'total_trades' in entry:
            trades_counts.append(entry['total_trades'])
    
    # Análise de portfolio
    if portfolio_values:
        print(f"💰 Portfolio: μ=${np.mean(portfolio_values):.2f}, σ=${np.std(portfolio_values):.2f}")
        print(f"    Range: [${min(portfolio_values):.2f}, ${max(portfolio_values):.2f}]")
        
        # Return sobre valor inicial (assumindo inicial = 500)
        initial_value = 500.0
        if portfolio_values:
            final_return = (portfolio_values[-1] - initial_value) / initial_value * 100
            best_return = (max(portfolio_values) - initial_value) / initial_value * 100
            print(f"    Return atual: {final_return:+.1f}%, Melhor: {best_return:+.1f}%")
    
    # Win rate
    if win_rates:
        print(f"🎯 Win Rate: μ={np.mean(win_rates)*100:.1f}%, σ={np.std(win_rates)*100:.1f}%")
        print(f"    Range: [{min(win_rates)*100:.1f}%, {max(win_rates)*100:.1f}%]")
    
    # Drawdown
    if drawdowns:
        print(f"📉 Max Drawdown: μ={np.mean(drawdowns)*100:.1f}%, σ={np.std(drawdowns)*100:.1f}%")
        print(f"    Range: [{min(drawdowns)*100:.1f}%, {max(drawdowns)*100:.1f}%]")
    
    # Trades
    if trades_counts:
        print(f"📊 Trades: μ={np.mean(trades_counts):.1f}, σ={np.std(trades_counts):.1f}")
        print(f"    Range: [{min(trades_counts)}, {max(trades_counts)}]")

def analyze_training_data(data, filename):
    """🚀 Analisa dados gerais de treinamento"""
    if not data:
        return
    
    print(f"\n🚀 ANÁLISE TREINAMENTO: {filename}")
    print("=" * 50)
    
    total_entries = len(data)
    print(f"📊 Total de entradas: {total_entries:,}")
    
    # Contar tipos de eventos
    event_types = {}
    phases = set()
    
    for entry in data:
        if 'event_type' in entry:
            event_type = entry['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        if 'phase' in entry:
            phases.add(entry['phase'])
    
    # Mostrar tipos de eventos
    if event_types:
        print(f"📋 Tipos de eventos:")
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {event_type}: {count:,} ({count/total_entries*100:.1f}%)")
    
    # Mostrar fases
    if phases:
        print(f"🎯 Fases detectadas: {', '.join(sorted(phases))}")
    
    # Estatísticas por campos comuns
    numeric_fields = ['step', 'episode_reward', 'episode_length', 'learning_rate']
    for field in numeric_fields:
        values = [entry[field] for entry in data if field in entry and isinstance(entry[field], (int, float))]
        if values:
            print(f"📊 {field}: μ={np.mean(values):.4f}, range=[{min(values):.4f}, {max(values):.4f}]")

def main():
    """🚀 Função principal"""
    print("🔍 ANALISADOR DE LOGS JSONL - Treinamento SILUS")
    print("=" * 60)
    
    # Encontrar arquivos JSONL
    jsonl_files = find_jsonl_files()
    
    if not jsonl_files:
        return
    
    print(f"\n📊 INICIANDO ANÁLISE DE {len(jsonl_files)} ARQUIVOS...")
    
    # Analisar cada arquivo
    for i, filepath in enumerate(jsonl_files[:5], 1):  # Limitar a 5 arquivos mais recentes
        filename = os.path.basename(filepath)
        print(f"\n{'='*60}")
        print(f"📁 ARQUIVO {i}: {filename}")
        print(f"📁 Caminho: {filepath}")
        print(f"📊 Tamanho: {os.path.getsize(filepath)/1024:.1f}KB")
        print(f"🕐 Modificado: {datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Carregar dados
        data = load_jsonl_data(filepath)
        
        if not data:
            print("⚠️ Nenhum dado válido encontrado")
            continue
        
        # Analisar baseado no tipo de arquivo
        filename_lower = filename.lower()
        
        if 'convergence' in filename_lower:
            analyze_convergence_data(data, filename)
        elif 'performance' in filename_lower:
            analyze_performance_data(data, filename)
        elif 'training' in filename_lower:
            analyze_training_data(data, filename)
        elif 'rewards' in filename_lower:
            analyze_convergence_data(data, filename)  # Rewards similar to convergence
        elif 'gradients' in filename_lower:
            analyze_convergence_data(data, filename)  # Gradients similar to convergence
        else:
            # Análise genérica
            print(f"\n🔍 ANÁLISE GENÉRICA: {filename}")
            print("=" * 50)
            print(f"📊 Total de entradas: {len(data):,}")
            
            if data:
                # Mostrar campos disponíveis
                all_fields = set()
                for entry in data[:100]:  # Amostra dos primeiros 100
                    all_fields.update(entry.keys())
                
                print(f"📋 Campos disponíveis: {', '.join(sorted(all_fields))}")
                
                # Mostrar amostra
                print(f"📋 Amostra (3 primeiras entradas):")
                for j, entry in enumerate(data[:3], 1):
                    print(f"  {j}. {json.dumps(entry, indent=None)[:150]}...")
    
    print(f"\n✅ ANÁLISE COMPLETA!")
    print(f"📊 {len(jsonl_files)} arquivos processados")

if __name__ == "__main__":
    main()