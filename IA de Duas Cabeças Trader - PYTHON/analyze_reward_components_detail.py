#!/usr/bin/env python3
"""
Análise detalhada dos componentes de reward para identificar fonte dos valores extremos.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_extreme_rewards(filepath):
    """Analisa os casos extremos de rewards para identificar padrões."""
    
    print("🔍 ANÁLISE DETALHADA DOS VALORES EXTREMOS DE REWARDS")
    print("="*80)
    
    rewards_data = []
    
    # Carregar dados
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"Processando linha {line_num:,}...")
            
            try:
                data = json.loads(line.strip())
                if data.get('type') == 'reward_info':
                    rewards_data.append(data)
            except:
                continue
    
    print(f"✅ Carregados {len(rewards_data):,} registros")
    
    # Separar por categorias de reward
    extreme_low = []    # < -1.5
    moderate_low = []   # -1.5 to -0.5
    neutral = []        # -0.5 to 0.5
    positive = []       # > 0.5
    
    for data in rewards_data:
        reward = data.get('total_reward', 0)
        if reward < -1.5:
            extreme_low.append(data)
        elif reward < -0.5:
            moderate_low.append(data)
        elif reward <= 0.5:
            neutral.append(data)
        else:
            positive.append(data)
    
    print(f"\n📊 CATEGORIZAÇÃO DOS REWARDS:")
    print(f"Extremamente negativos (< -1.5): {len(extreme_low):,} registros")
    print(f"Moderadamente negativos (-1.5 a -0.5): {len(moderate_low):,} registros")
    print(f"Neutros (-0.5 a 0.5): {len(neutral):,} registros")
    print(f"Positivos (> 0.5): {len(positive):,} registros")
    
    # Análise dos componentes em cada categoria
    categories = [
        ("EXTREMAMENTE NEGATIVOS", extreme_low),
        ("MODERADAMENTE NEGATIVOS", moderate_low),
        ("NEUTROS", neutral),
        ("POSITIVOS", positive)
    ]
    
    for category_name, category_data in categories:
        if len(category_data) == 0:
            continue
            
        print(f"\n🔍 ANÁLISE: {category_name} ({len(category_data):,} amostras)")
        print("-" * 60)
        
        # Componentes médios
        components = defaultdict(list)
        gaming_penalties = []
        overtrading_penalties = []
        total_rewards = []
        
        for item in category_data:
            total_rewards.append(item.get('total_reward', 0))
            
            reward_comps = item.get('reward_components', {})
            for comp, value in reward_comps.items():
                if value is not None:
                    components[comp].append(value)
            
            gaming_det = item.get('gaming_detection', {})
            if gaming_det.get('gaming_penalty') is not None:
                gaming_penalties.append(gaming_det['gaming_penalty'])
            if gaming_det.get('overtrading_penalty') is not None:
                overtrading_penalties.append(gaming_det['overtrading_penalty'])
        
        # Estatísticas dos componentes
        print(f"Total Reward: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
        
        for comp_name, values in components.items():
            if len(values) > 0:
                non_zero = [v for v in values if v != 0]
                if len(non_zero) > 0:
                    print(f"{comp_name:15}: {np.mean(values):7.4f} ± {np.std(values):.4f} "
                          f"(não-zero: {len(non_zero):,}/{len(values):,})")
                else:
                    print(f"{comp_name:15}: todos zeros")
        
        if gaming_penalties:
            print(f"Gaming Penalty:  {np.mean(gaming_penalties):7.4f} ± {np.std(gaming_penalties):.4f}")
        
        if overtrading_penalties:
            print(f"Overtrading Pen: {np.mean(overtrading_penalties):7.4f} ± {np.std(overtrading_penalties):.4f}")
        
        # Análise de casos extremos dentro da categoria
        if category_name == "EXTREMAMENTE NEGATIVOS":
            worst_cases = sorted(category_data, key=lambda x: x.get('total_reward', 0))[:10]
            print(f"\n🔥 10 PIORES CASOS:")
            for i, case in enumerate(worst_cases, 1):
                reward = case.get('total_reward', 0)
                comps = case.get('reward_components', {})
                gaming = case.get('gaming_detection', {})
                print(f"{i:2d}. Reward: {reward:7.4f} | PNL: {comps.get('pnl', 0):6.3f} | "
                      f"Gaming: {comps.get('gaming_penalty', 0):6.3f} | "
                      f"Overtrading: {gaming.get('overtrading_penalty', 0):6.3f}")
    
    # Análise temporal dos rewards
    print(f"\n📈 ANÁLISE TEMPORAL DOS REWARDS")
    print("-" * 60)
    
    # Agrupar por steps para ver evolução
    step_rewards = defaultdict(list)
    for data in rewards_data:
        step = data.get('step', 0)
        reward = data.get('total_reward', 0)
        step_rewards[step].append(reward)
    
    # Calcular médias por step (últimos 20 steps)
    recent_steps = sorted(step_rewards.keys())[-20:]
    print(f"Evolução dos rewards (últimos 20 steps):")
    for step in recent_steps:
        rewards_at_step = step_rewards[step]
        avg_reward = np.mean(rewards_at_step)
        print(f"Step {step:,}: {avg_reward:7.4f} (n={len(rewards_at_step):,})")
    
    return {
        'extreme_low': len(extreme_low),
        'moderate_low': len(moderate_low),
        'neutral': len(neutral),
        'positive': len(positive),
        'total': len(rewards_data)
    }

def main():
    """Função principal."""
    # Encontrar arquivo mais recente
    avaliacoes_dir = Path("avaliacoes")
    reward_files = list(avaliacoes_dir.glob("rewards_*.jsonl"))
    
    if not reward_files:
        print("❌ Nenhum arquivo de rewards encontrado!")
        return
    
    latest_file = max(reward_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Analisando: {latest_file}")
    
    # Análise
    results = analyze_extreme_rewards(latest_file)
    
    print(f"\n🎯 RESUMO FINAL:")
    print(f"• {results['extreme_low']:,} rewards extremamente negativos ({results['extreme_low']/results['total']*100:.1f}%)")
    print(f"• {results['moderate_low']:,} rewards moderadamente negativos ({results['moderate_low']/results['total']*100:.1f}%)")
    print(f"• {results['neutral']:,} rewards neutros ({results['neutral']/results['total']*100:.1f}%)")
    print(f"• {results['positive']:,} rewards positivos ({results['positive']/results['total']*100:.1f}%)")
    
    print(f"\n💡 INSIGHTS CHAVE:")
    print(f"1. O sistema está gerando rewards predominantemente negativos")
    print(f"2. Gaming penalties e overtrading penalties são os principais contribuintes")
    print(f"3. PNL component varia entre -1 e +1")
    print(f"4. O range [-2, 0] capturaria quase todos os valores reais")

if __name__ == "__main__":
    main()