"""
🕵️ QUICK REWARD HACKING CHECK
Análise pontual dos logs JSONL para detectar reward hacking
"""

import json
import glob
import numpy as np
from datetime import datetime
import os

def analyze_latest_jsonl():
    """Analisa os arquivos JSONL mais recentes"""
    
    print("🕵️ ANÁLISE DE REWARD HACKING - V5 SHARPE")
    print("="*60)
    
    # Buscar arquivos mais recentes
    patterns = {
        'training': 'avaliacoes/training_*.jsonl',
        'rewards': 'avaliacoes/rewards_*.jsonl',
        'performance': 'avaliacoes/performance_*.jsonl'
    }
    
    latest_files = {}
    for name, pattern in patterns.items():
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=os.path.getmtime)
            latest_files[name] = latest
            print(f"📂 {name}: {os.path.basename(latest)}")
    
    if not latest_files:
        print("❌ Nenhum arquivo JSONL encontrado!")
        return
    
    # Análise por tipo
    results = {}
    
    # 1. TRAINING DATA (Clip Fraction)
    if 'training' in latest_files:
        print(f"\n📊 ANALISANDO TRAINING DATA...")
        training_data = read_jsonl(latest_files['training'], max_lines=100)
        
        clip_fractions = []
        steps = []
        
        for entry in training_data:
            if 'clip_fraction' in entry:
                clip_fractions.append(entry['clip_fraction'])
            if 'step' in entry:
                steps.append(entry['step'])
        
        if clip_fractions:
            avg_clip = np.mean(clip_fractions[-20:])  # Últimas 20
            recent_clip = np.mean(clip_fractions[-5:])  # Últimas 5
            
            results['clip_fraction'] = {
                'avg': avg_clip,
                'recent': recent_clip,
                'count': len(clip_fractions)
            }
            
            print(f"✂️ Clip Fraction Médio (últimas 20): {avg_clip:.3f}")
            print(f"✂️ Clip Fraction Recente (últimas 5): {recent_clip:.3f}")
            
            if avg_clip > 0.35:
                print("🚨 ALERTA: Clip fraction muito alto! Possível instabilidade")
            elif avg_clip > 0.25:
                print("⚠️ ATENÇÃO: Clip fraction elevado - monitorar")
            else:
                print("✅ Clip fraction normal")
        
        if steps:
            print(f"📈 Steps processados: {max(steps):,}")
    
    # 2. REWARDS DATA
    if 'rewards' in latest_files:
        print(f"\n🎯 ANALISANDO REWARDS DATA...")
        rewards_data = read_jsonl(latest_files['rewards'], max_lines=100)
        
        episode_rewards = []
        reward_values = []
        
        for entry in rewards_data:
            if 'episode_reward_mean' in entry:
                episode_rewards.append(entry['episode_reward_mean'])
            elif 'reward' in entry:
                reward_values.append(entry['reward'])
        
        # Usar episode_rewards se disponível, senão reward_values
        rewards = episode_rewards if episode_rewards else reward_values
        
        if rewards and len(rewards) >= 10:
            avg_reward = np.mean(rewards[-20:])  # Últimas 20
            recent_reward = np.mean(rewards[-5:])   # Últimas 5
            reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) >= 5 else 0
            
            results['rewards'] = {
                'avg': avg_reward,
                'recent': recent_reward,
                'trend': reward_trend,
                'count': len(rewards)
            }
            
            print(f"🎯 Reward Médio (últimas 20): {avg_reward:.4f}")
            print(f"🎯 Reward Recente (últimas 5): {recent_reward:.4f}")
            print(f"📈 Tendência: {'+' if reward_trend > 0 else ''}{reward_trend:.6f}")
            
            if reward_trend > 0.01:
                print("📈 Rewards crescendo - pode ser bom sinal ou hacking")
            elif reward_trend < -0.01:
                print("📉 Rewards decrescendo - possível instabilidade")
            else:
                print("➡️ Rewards estáveis")
    
    # 3. PERFORMANCE DATA (Real Trading Performance)
    if 'performance' in latest_files:
        print(f"\n💰 ANALISANDO PERFORMANCE DATA...")
        performance_data = read_jsonl(latest_files['performance'], max_lines=100)
        
        portfolio_values = []
        
        for entry in performance_data:
            if 'portfolio_value' in entry:
                portfolio_values.append(entry['portfolio_value'])
            elif 'total_portfolio_value' in entry:
                portfolio_values.append(entry['total_portfolio_value'])
        
        if portfolio_values and len(portfolio_values) >= 20:
            initial_portfolio = portfolio_values[0] if portfolio_values[0] > 0 else 1000.0
            final_portfolio = portfolio_values[-1]
            
            # Calcular returns para Sharpe
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = returns[~np.isnan(returns)]  # Remove NaNs
            
            growth_pct = (final_portfolio / initial_portfolio - 1) * 100
            
            # Sharpe ratio estimado
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_estimate = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Anualizado
            else:
                sharpe_estimate = 0.0
            
            results['performance'] = {
                'growth': growth_pct,
                'sharpe': sharpe_estimate,
                'initial': initial_portfolio,
                'final': final_portfolio
            }
            
            print(f"💰 Portfolio: ${initial_portfolio:.2f} → ${final_portfolio:.2f}")
            print(f"📈 Crescimento: {growth_pct:.2f}%")
            print(f"📊 Sharpe Estimado: {sharpe_estimate:.3f}")
            
            if sharpe_estimate > 1.0:
                print("🎯 EXCELENTE: Sharpe > 1.0 - performance institucional!")
            elif sharpe_estimate > 0.5:
                print("✅ BOM: Sharpe > 0.5 - performance positiva")
            elif sharpe_estimate > 0.0:
                print("🟡 MEDIANO: Sharpe > 0 - melhorar consistency")
            else:
                print("🔴 RUIM: Sharpe negativo - revisar estratégia")
    
    # 4. ANÁLISE CRUZADA - DETECTAR REWARD HACKING
    print(f"\n🔍 ANÁLISE DE REWARD HACKING:")
    print("-"*40)
    
    hacking_alerts = []
    
    # Clip fraction muito alto persistente
    if 'clip_fraction' in results and results['clip_fraction']['avg'] > 0.35:
        hacking_alerts.append("Clip fraction persistentemente alto")
    
    # Rewards crescendo mas performance não
    if ('rewards' in results and 'performance' in results):
        reward_positive = results['rewards']['trend'] > 0.01
        performance_stagnant = abs(results['performance']['growth']) < 1.0  # Menos de 1%
        
        if reward_positive and performance_stagnant:
            hacking_alerts.append("Rewards crescendo mas portfolio estagnado")
        
        # Sharpe baixo com rewards altos
        if results['rewards']['avg'] > 0.5 and results['performance']['sharpe'] < 0.2:
            hacking_alerts.append("Rewards altos mas Sharpe ratio baixo")
    
    if hacking_alerts:
        print("🚨 SINAIS DE POSSÍVEL REWARD HACKING:")
        for alert in hacking_alerts:
            print(f"   ⚠️ {alert}")
        print("\n💡 RECOMENDAÇÕES:")
        print("   • Investigar reward function para exploits")
        print("   • Verificar se agent está maximizando métricas irrelevantes")
        print("   • Considerar ajustar reward weights ou adicionar constraints")
    else:
        print("✅ NENHUM SINAL CLARO DE REWARD HACKING DETECTADO")
        print("📊 Sistema aparenta estar treinando normalmente")
    
    # Timestamp
    print(f"\n⏰ Análise realizada em: {datetime.now().strftime('%H:%M:%S')}")

def read_jsonl(file_path, max_lines=200):
    """Lê arquivo JSONL com limite de linhas"""
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Pegar últimas linhas
        recent_lines = lines[-max_lines:] if len(lines) > max_lines else lines
        
        for line in recent_lines:
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except json.JSONDecodeError:
                continue
                
        return data
    except Exception as e:
        print(f"⚠️ Erro lendo {file_path}: {e}")
        return []

if __name__ == "__main__":
    try:
        analyze_latest_jsonl()
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        import traceback
        traceback.print_exc()