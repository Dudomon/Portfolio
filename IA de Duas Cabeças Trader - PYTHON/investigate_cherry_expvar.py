"""
🔍 INVESTIGAÇÃO DE EXPLAINED VARIANCE NEGATIVA - CHERRY
Analisa possíveis problemas no reward system que causam EV negativa
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Adicionar paths
sys.path.append(r'D:\Projeto')

print("🔍 [INVESTIGAÇÃO] Analisando problemas de Explained Variance no Cherry...")

try:
    # Importar reward system
    from trading_framework.rewards.reward_system_simple import create_simple_reward_system
    from cherry import REALISTIC_SLTP_CONFIG, convert_action_to_realistic_sltp
    print("✅ Imports realizados com sucesso")
    
except ImportError as e:
    print(f"❌ Erro na importação: {e}")
    sys.exit(1)

class DetailedMockEnv:
    """Mock environment para análise detalhada"""
    def __init__(self):
        self.trades = []
        self.positions = []
        self.current_step = 100
        self.current_drawdown = 0.0
        self.df = None
        self.balance_history = [500.0]  # Histórico de balance
        
    def add_trade(self, pnl_usd, sl_points, tp_points, duration_steps=30, trade_type='long'):
        """Adicionar trade com tracking de balance"""
        trade = {
            'pnl_usd': pnl_usd,
            'sl_points': sl_points,
            'tp_points': tp_points,
            'duration_steps': duration_steps,
            'type': trade_type,
            'entry_price': 2000.0,
            'exit_price': 2000.0 + (pnl_usd / 0.02 / 100),
            'sl_price': 2000.0 - sl_points if trade_type == 'long' else 2000.0 + sl_points,
            'tp_price': 2000.0 + tp_points if trade_type == 'long' else 2000.0 - tp_points,
            'exit_reason': 'tp' if pnl_usd > 0 else 'sl'
        }
        self.trades.append(trade)
        
        # Atualizar balance
        new_balance = self.balance_history[-1] + pnl_usd
        self.balance_history.append(new_balance)
        
        # Calcular drawdown atual
        peak_balance = max(self.balance_history)
        if peak_balance > 0:
            self.current_drawdown = ((peak_balance - new_balance) / peak_balance) * 100
        
        return trade

def analyze_reward_distribution():
    """Analisar distribuição de rewards para detectar problemas"""
    print("\n🔍 [ANÁLISE 1] Distribuição de Rewards...")
    
    reward_system = create_simple_reward_system(500.0)
    env = DetailedMockEnv()
    
    # Cenários realistas de trading
    realistic_scenarios = [
        # Wins pequenos (mais comuns)
        (3.0, 10, 15, 25, "Win pequeno"),
        (5.0, 12, 18, 35, "Win médio pequeno"),
        (8.0, 15, 22, 45, "Win médio"),
        
        # Wins grandes (menos comuns)
        (12.0, 18, 28, 60, "Win grande"),
        (20.0, 20, 35, 80, "Win muito grande"),
        
        # Losses pequenos (comuns)
        (-4.0, 8, 12, 20, "Loss pequeno"),
        (-6.0, 10, 15, 30, "Loss médio pequeno"),
        (-8.0, 12, 18, 25, "Loss médio"),
        
        # Losses grandes (menos comuns mas impactantes)
        (-12.0, 15, 25, 40, "Loss grande"),
        (-18.0, 20, 30, 50, "Loss muito grande"),
        
        # Breakeven/micro results
        (0.5, 8, 10, 15, "Breakeven positivo"),
        (-0.5, 8, 10, 20, "Breakeven negativo"),
        (1.0, 25, 40, 180, "Micro win longo"),
    ]
    
    rewards = []
    reward_components = defaultdict(list)
    
    print("   Cenário                    | PnL    | Reward  | PnL Component | Win/Loss | Completion")
    print("   " + "-" * 85)
    
    for pnl, sl, tp, duration, desc in realistic_scenarios:
        # Resetar para cada cenário
        test_env = DetailedMockEnv()
        old_trades_count = len(test_env.trades)
        
        test_env.add_trade(pnl, sl, tp, duration)
        
        action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        old_state = {'trades_count': old_trades_count}
        
        reward, info, done = reward_system.calculate_reward_and_info(test_env, action, old_state)
        
        rewards.append(reward)
        
        # Extrair componentes principais
        pnl_component = info['components'].get('pnl_direct', 0)
        win_loss_component = info['components'].get('win_bonus', info['components'].get('loss_penalty', 0))
        completion_bonus = info['components'].get('trade_completion_bonus', 0)
        
        reward_components['pnl'].append(pnl_component)
        reward_components['win_loss'].append(win_loss_component)
        reward_components['completion'].append(completion_bonus)
        
        print(f"   {desc:25} | ${pnl:5.1f} | {reward:7.2f} | {pnl_component:11.2f} | {win_loss_component:8.2f} | {completion_bonus:9.2f}")
    
    # Análise estatística
    rewards_array = np.array(rewards)
    print(f"\n📊 ESTATÍSTICAS DE REWARDS:")
    print(f"   Média: {np.mean(rewards_array):.2f}")
    print(f"   Mediana: {np.median(rewards_array):.2f}")
    print(f"   Std Dev: {np.std(rewards_array):.2f}")
    print(f"   Min: {np.min(rewards_array):.2f}")
    print(f"   Max: {np.max(rewards_array):.2f}")
    print(f"   Range: {np.max(rewards_array) - np.min(rewards_array):.2f}")
    
    # 🚨 PROBLEMA IDENTIFICADO 1: Rewards muito altos
    if np.max(rewards_array) > 50:
        print(f"🚨 PROBLEMA: Rewards muito altos (max: {np.max(rewards_array):.2f})")
        print("   → Pode causar instabilidade no treinamento")
    
    # 🚨 PROBLEMA IDENTIFICADO 2: Assimetria excessiva
    positive_rewards = rewards_array[rewards_array > 0]
    negative_rewards = rewards_array[rewards_array < 0]
    
    if len(positive_rewards) > 0 and len(negative_rewards) > 0:
        pos_mean = np.mean(positive_rewards)
        neg_mean = np.mean(negative_rewards)
        asymmetry_ratio = abs(pos_mean / neg_mean) if neg_mean != 0 else float('inf')
        
        print(f"\n📊 ASSIMETRIA:")
        print(f"   Rewards positivos: média {pos_mean:.2f}")
        print(f"   Rewards negativos: média {neg_mean:.2f}")
        print(f"   Razão de assimetria: {asymmetry_ratio:.2f}")
        
        if asymmetry_ratio > 3.0 or asymmetry_ratio < 0.33:
            print(f"🚨 PROBLEMA: Assimetria excessiva (razão: {asymmetry_ratio:.2f})")
            print("   → Modelo pode ter bias para always hold ou always trade")
    
    return rewards_array, reward_components

def analyze_reward_clipping():
    """Analisar problemas de clipping de rewards"""
    print("\n🔍 [ANÁLISE 2] Problemas de Clipping...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Testar valores extremos
    extreme_scenarios = [
        (50.0, 8, 12, 30, "Win extremo"),
        (-50.0, 25, 40, 60, "Loss extremo"),
        (100.0, 15, 25, 45, "Win absurdo"),
        (-100.0, 20, 35, 90, "Loss absurdo"),
    ]
    
    print("   Cenário        | PnL     | Reward Raw | Reward Final | Clipped?")
    print("   " + "-" * 65)
    
    clipping_issues = 0
    
    for pnl, sl, tp, duration, desc in extreme_scenarios:
        env = DetailedMockEnv()
        env.add_trade(pnl, sl, tp, duration)
        
        action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        old_state = {'trades_count': 0}
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)
        
        # O reward está sendo clippado em [-100, 100] no código
        clipped = abs(reward) >= 99.9  # Próximo dos limites
        if clipped:
            clipping_issues += 1
            
        print(f"   {desc:13} | ${pnl:6.1f} | {reward:10.2f} | {reward:11.2f} | {'SIM' if clipped else 'NÃO'}")
    
    if clipping_issues > 0:
        print(f"\n🚨 PROBLEMA: {clipping_issues} casos de clipping detectados")
        print("   → Clipping pode causar perda de gradiente")
    
    return clipping_issues

def analyze_reward_sparsity():
    """Analisar esparsidade de rewards (muitos zeros)"""
    print("\n🔍 [ANÁLISE 3] Esparsidade de Rewards...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Simular sequência típica de steps (maioria HOLD)
    hold_rewards = []
    trade_rewards = []
    
    # Simular 100 steps, sendo 90 HOLD e 10 com trades
    for step in range(100):
        env = DetailedMockEnv()
        
        if step % 10 == 0:  # A cada 10 steps, um trade
            env.add_trade(np.random.normal(2.0, 8.0), 12, 20, 30)  # Trade aleatório
            action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            old_state = {'trades_count': 0}
            trade_rewards.append(True)
        else:  # HOLD action
            action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            old_state = {'trades_count': len(env.trades)}
            trade_rewards.append(False)
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)
        hold_rewards.append(reward)
    
    # Analisar esparsidade
    zero_rewards = sum(1 for r in hold_rewards if abs(r) < 0.01)
    sparse_ratio = zero_rewards / len(hold_rewards)
    
    print(f"   Total steps analisados: {len(hold_rewards)}")
    print(f"   Steps com reward ~0: {zero_rewards}")
    print(f"   Razão de esparsidade: {sparse_ratio:.2%}")
    
    if sparse_ratio > 0.7:
        print(f"🚨 PROBLEMA: Alta esparsidade ({sparse_ratio:.2%})")
        print("   → Modelo pode ter dificuldade de aprender com poucos sinais")
    
    # Analisar variância dentro dos holds
    hold_only_rewards = [hold_rewards[i] for i in range(len(hold_rewards)) if not trade_rewards[i]]
    if len(hold_only_rewards) > 1:
        hold_variance = np.var(hold_only_rewards)
        print(f"   Variância em HOLD rewards: {hold_variance:.4f}")
        
        if hold_variance < 0.001:
            print(f"🚨 PROBLEMA: Variância muito baixa em HOLDs ({hold_variance:.4f})")
            print("   → Modelo pode não conseguir distinguir bons vs maus HOLDs")
    
    return sparse_ratio, hold_only_rewards

def analyze_reward_scaling():
    """Analisar escala inadequada de rewards"""
    print("\n🔍 [ANÁLISE 4] Problemas de Escala...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Comparar magnitudes dos componentes
    env = DetailedMockEnv()
    env.add_trade(10.0, 12, 20, 40)  # Trade médio
    
    action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
    old_state = {'trades_count': 0}
    
    reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)
    
    print("   Componente                 | Valor     | Peso Relativo")
    print("   " + "-" * 55)
    
    components = info['components']
    total_magnitude = sum(abs(v) for v in components.values())
    
    scaling_issues = 0
    
    for comp_name, comp_value in components.items():
        if total_magnitude > 0:
            relative_weight = (abs(comp_value) / total_magnitude) * 100
        else:
            relative_weight = 0
            
        print(f"   {comp_name:25} | {comp_value:8.2f} | {relative_weight:6.1f}%")
        
        # Detectar problemas de escala
        if abs(comp_value) > 50:  # Componente muito alto
            scaling_issues += 1
            print(f"      🚨 Componente muito alto: {comp_value:.2f}")
            
        if relative_weight > 80:  # Um componente domina demais
            scaling_issues += 1
            print(f"      🚨 Componente dominante: {relative_weight:.1f}%")
    
    if scaling_issues > 0:
        print(f"\n🚨 PROBLEMAS DE ESCALA: {scaling_issues} detectados")
        print("   → Componentes desbalanceados podem causar instabilidade")
    
    return scaling_issues, components

def identify_reward_pathologies():
    """Identificar patologias específicas do reward system"""
    print("\n🔍 [ANÁLISE 5] Patologias do Reward System...")
    
    pathologies = []
    
    # Patologia 1: PnL direto muito alto
    reward_system = create_simple_reward_system(500.0)
    if hasattr(reward_system, 'weights'):
        pnl_weight = reward_system.weights.get('pnl_direct', 0)
        if pnl_weight > 10:
            pathologies.append(f"PnL direto muito alto: {pnl_weight}")
            print(f"🚨 PATOLOGIA: PnL weight = {pnl_weight} (muito alto)")
    
    # Patologia 2: Testar reward explosion
    env = DetailedMockEnv()
    env.add_trade(25.0, 10, 15, 30)  # Win grande
    
    action = np.array([1.0, 0.9, 0.0, 0.0, 0.0, 0.0])
    old_state = {'trades_count': 0}
    
    reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)
    
    if abs(reward) > 200:
        pathologies.append(f"Reward explosion: {reward}")
        print(f"🚨 PATOLOGIA: Reward muito alto: {reward}")
    
    # Patologia 3: Testar progressive risk zones
    env_high_dd = DetailedMockEnv()
    env_high_dd.current_drawdown = 25.0  # DD alto
    env_high_dd.add_trade(5.0, 12, 18, 35)
    
    reward_dd, info_dd, _ = reward_system.calculate_reward_and_info(env_high_dd, action, old_state)
    
    if 'target_analysis' in info_dd and 'enhanced_risk_v7' in info_dd['target_analysis']:
        risk_info = info_dd['target_analysis']['enhanced_risk_v7']
        if 'progressive_zones' in risk_info:
            zone_penalty = risk_info['progressive_zones'].get('weighted_reward', 0)
            if abs(zone_penalty) > 10:
                pathologies.append(f"Risk zone penalty muito alto: {zone_penalty}")
                print(f"🚨 PATOLOGIA: Risk penalty muito alto: {zone_penalty}")
    
    return pathologies

def suggest_fixes():
    """Sugerir correções para os problemas encontrados"""
    print("\n💡 [SOLUÇÕES] Correções Sugeridas...")
    
    print("1. 🎯 REDUZIR MAGNITUDE DOS REWARDS:")
    print("   - pnl_direct: 15.0 → 3.0 (5x menor)")
    print("   - win_bonus: 8.0 → 2.0 (4x menor)")  
    print("   - loss_penalty: -5.0 → -1.5 (3x menor)")
    print("")
    
    print("2. 🎯 MELHORAR ESPARSIDADE:")
    print("   - Adicionar small_reward para HOLDs bem executados (+0.1)")
    print("   - Reward por tempo em posição sem loss (-0.05/step)")
    print("   - Micro rewards por análise técnica correta")
    print("")
    
    print("3. 🎯 AJUSTAR CLIPPING:")
    print("   - Limites: [-100, 100] → [-20, 20]")
    print("   - Usar soft clipping (tanh) ao invés de hard clipping")
    print("")
    
    print("4. 🎯 BALANCEAR COMPONENTES:")
    print("   - PnL: 60% → 40%")
    print("   - SL/TP: 25% → 20%")
    print("   - Atividade: 10% → 20%")
    print("   - Risco: 5% → 10%")
    print("   - Novo Dense Rewards: 0% → 10%")
    print("")
    
    print("5. 🎯 PROGRESSIVE RISK SUAVIZADO:")
    print("   - Penalidades: [-2.0, -0.8, -0.3] → [-0.5, -0.2, -0.1]")
    print("   - Multiplicadores: [0.6, 0.8, 0.9] → [0.9, 0.95, 0.98]")

def main():
    print("🍒 ===============================================")
    print("🍒 INVESTIGAÇÃO EXPLAINED VARIANCE - CHERRY")  
    print("🍒 ===============================================")
    
    # Executar análises
    rewards_dist, components = analyze_reward_distribution()
    clipping_issues = analyze_reward_clipping()
    sparse_ratio, hold_rewards = analyze_reward_sparsity()
    scaling_issues, comp_breakdown = analyze_reward_scaling()
    pathologies = identify_reward_pathologies()
    
    # Resumo dos problemas
    print("\n" + "="*60)
    print("📋 RESUMO DOS PROBLEMAS ENCONTRADOS")
    print("="*60)
    
    total_issues = 0
    
    if np.max(rewards_dist) > 50:
        print("❌ Rewards muito altos (instabilidade)")
        total_issues += 1
    
    if clipping_issues > 0:
        print(f"❌ {clipping_issues} casos de clipping (perda de gradiente)")
        total_issues += 1
    
    if sparse_ratio > 0.7:
        print(f"❌ Alta esparsidade ({sparse_ratio:.1%}) - poucos sinais")
        total_issues += 1
    
    if scaling_issues > 0:
        print(f"❌ {scaling_issues} problemas de escala")
        total_issues += 1
    
    if len(pathologies) > 0:
        print(f"❌ {len(pathologies)} patologias detectadas")
        total_issues += 1
    
    print(f"\n🎯 TOTAL DE PROBLEMAS: {total_issues}")
    
    if total_issues >= 3:
        print("🚨 DIAGNÓSTICO: REWARD SYSTEM PROBLEMÁTICO")
        print("   → Explained Variance negativa é ESPERADA com estes problemas")
    elif total_issues >= 2:
        print("⚠️ DIAGNÓSTICO: REWARD SYSTEM COM PROBLEMAS MODERADOS")
    else:
        print("✅ DIAGNÓSTICO: REWARD SYSTEM RELATIVAMENTE SAUDÁVEL")
    
    # Sugerir correções
    suggest_fixes()
    
    print(f"\n🎯 PRÓXIMO PASSO: Implementar correções no reward system")

if __name__ == "__main__":
    main()