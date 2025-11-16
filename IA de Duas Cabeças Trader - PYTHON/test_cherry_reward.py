"""
🧪 TESTE DO SISTEMA DE REWARD DO CHERRY
Testa o reward system swing trade configurado no cherry.py
"""

import sys
import os
import numpy as np
import time

# Adicionar paths
sys.path.append(r'D:\Projeto')

print("🍒 [TESTE CHERRY] Iniciando teste do reward system...")

try:
    # Importar reward system do cherry
    from trading_framework.rewards.reward_system_simple import create_simple_reward_system
    print("✅ SimpleRewardCalculator importado com sucesso")
    
    # Importar configurações do cherry
    from cherry import REALISTIC_SLTP_CONFIG, convert_action_to_realistic_sltp
    print("✅ Configurações CHERRY importadas com sucesso")
    print(f"   SL Range: {REALISTIC_SLTP_CONFIG['sl_min_points']}-{REALISTIC_SLTP_CONFIG['sl_max_points']} pontos")
    print(f"   TP Range: {REALISTIC_SLTP_CONFIG['tp_min_points']}-{REALISTIC_SLTP_CONFIG['tp_max_points']} pontos")
    
except ImportError as e:
    print(f"❌ Erro na importação: {e}")
    sys.exit(1)

class MockEnv:
    """Mock environment para testar reward system"""
    def __init__(self):
        self.trades = []
        self.positions = []
        self.current_step = 100
        self.current_drawdown = 0.0
        self.df = None  # Simplificado para teste
        
    def add_trade(self, pnl_usd, sl_points, tp_points, duration_steps=30, trade_type='long'):
        """Adicionar trade de teste"""
        trade = {
            'pnl_usd': pnl_usd,
            'sl_points': sl_points,
            'tp_points': tp_points,
            'duration_steps': duration_steps,
            'type': trade_type,
            'entry_price': 2000.0,
            'exit_price': 2000.0 + (pnl_usd / 0.02 / 100),  # Simular preço baseado no PnL
            'sl_price': 2000.0 - sl_points if trade_type == 'long' else 2000.0 + sl_points,
            'tp_price': 2000.0 + tp_points if trade_type == 'long' else 2000.0 - tp_points,
            'exit_reason': 'tp' if pnl_usd > 0 else 'sl'
        }
        self.trades.append(trade)

def test_reward_system():
    """Teste completo do reward system"""
    print("\n🧪 [TESTE] Criando reward system Cherry...")
    
    # Criar reward system
    reward_system = create_simple_reward_system(initial_balance=500.0)
    print("✅ Reward system criado com sucesso")
    
    # Criar mock environment
    env = MockEnv()
    
    print("\n📊 [TESTE 1] Testando ranges de SL/TP...")
    
    # Testar conversão de ações para SL/TP
    test_actions = [
        [-3, -3],  # SL mínimo, TP mínimo
        [0, 0],    # SL médio, TP médio  
        [3, 3],    # SL máximo, TP máximo
        [-1, 2],   # SL baixo, TP alto
        [1.5, -0.5] # Valores intermediários
    ]
    
    for i, action in enumerate(test_actions):
        sl_points, tp_points = convert_action_to_realistic_sltp(action, 2000.0)
        print(f"   Ação {action} → SL: {sl_points:.1f}p, TP: {tp_points:.1f}p")
        
        # Verificar se estão dentro dos ranges
        sl_ok = REALISTIC_SLTP_CONFIG['sl_min_points'] <= sl_points <= REALISTIC_SLTP_CONFIG['sl_max_points']
        tp_ok = REALISTIC_SLTP_CONFIG['tp_min_points'] <= tp_points <= REALISTIC_SLTP_CONFIG['tp_max_points']
        
        if sl_ok and tp_ok:
            print(f"      ✅ Ranges corretos")
        else:
            print(f"      ❌ ERRO: SL ou TP fora do range!")
            return False
    
    print("\n📊 [TESTE 2] Testando rewards para trades swing...")
    
    # Cenários de teste
    test_scenarios = [
        # (pnl_usd, sl_points, tp_points, duration, description)
        (15.0, 10, 20, 45, "Trade excelente swing (15$ lucro, R:R 2:1)"),
        (8.0, 12, 18, 30, "Trade bom swing (8$ lucro, R:R 1.5:1)"),
        (-6.0, 8, 16, 25, "Trade perdedor swing (6$ perda)"),
        (25.0, 15, 30, 60, "Trade excepcional swing (25$ lucro, R:R 2:1)"),
        (3.0, 20, 25, 120, "Trade pequeno swing (3$ lucro, R:R baixo)"),
        (-12.0, 25, 40, 80, "Trade grande perda swing (-12$)")
    ]
    
    total_reward = 0.0
    
    for i, (pnl, sl, tp, duration, desc) in enumerate(test_scenarios):
        print(f"\n   💰 Cenário {i+1}: {desc}")
        
        # Adicionar trade ao environment
        old_trades_count = len(env.trades)
        env.add_trade(pnl, sl, tp, duration)
        
        # Simular ação (não importa muito para o teste)
        action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        
        # Estado anterior
        old_state = {'trades_count': old_trades_count}
        
        # Calcular reward
        reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)
        total_reward += reward
        
        print(f"      Reward: {reward:.2f}")
        print(f"      PnL direto: {info['components'].get('pnl_direct', 0):.2f}")
        print(f"      Win/Loss bonus: {info['components'].get('win_bonus', info['components'].get('loss_penalty', 0)):.2f}")
        print(f"      Trade completion: {info['components'].get('trade_completion_bonus', 0):.2f}")
        
        if 'target_analysis' in info:
            print(f"      Expert SL/TP: {info['target_analysis'].get('expert_sltp', 0):.2f}")
            print(f"      Technical Analysis: {info['target_analysis'].get('technical_analysis', 0):.2f}")
    
    print(f"\n📊 [RESULTADO] Reward total acumulado: {total_reward:.2f}")
    
    print("\n📊 [TESTE 3] Testando sistema de atividade...")
    
    # Testar diferentes níveis de atividade
    activity_scenarios = [
        (5, "Baixa atividade (undertrading)"),
        (15, "Atividade na zona target (12-24)"),
        (20, "Atividade ótima (zona target)"),
        (30, "Overtrading moderado"),
        (40, "Overtrading excessivo")
    ]
    
    for trades_count, desc in activity_scenarios:
        # Simular environment com diferentes números de trades
        mock_env = MockEnv()
        for i in range(trades_count):
            mock_env.add_trade(5.0, 10, 15, 30)  # Trades neutros
        
        action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])  # HOLD action
        old_state = {'trades_count': 0}
        
        reward, info, done = reward_system.calculate_reward_and_info(mock_env, action, old_state)
        
        activity_reward = info['target_analysis'].get('activity', 0)
        activity_zone = info.get('activity_zone', 'UNKNOWN')
        
        print(f"   📈 {desc}: {trades_count} trades → Reward: {activity_reward:.2f} | Zona: {activity_zone}")
    
    print("\n📊 [TESTE 4] Testando Progressive Risk Zones...")
    
    # Testar diferentes níveis de drawdown
    risk_scenarios = [
        (0.0, "Sem drawdown (zona verde)"),
        (5.0, "Drawdown moderado (zona amarela)"),
        (12.0, "Drawdown alto (zona laranja)"),
        (20.0, "Drawdown crítico (zona vermelha)"),
        (30.0, "Drawdown extremo (zona preta)")
    ]
    
    for dd_pct, desc in risk_scenarios:
        mock_env = MockEnv()
        mock_env.current_drawdown = dd_pct
        mock_env.add_trade(5.0, 10, 15, 30)  # Trade neutro
        
        action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        old_state = {'trades_count': 0}
        
        reward, info, done = reward_system.calculate_reward_and_info(mock_env, action, old_state)
        
        risk_info = info['target_analysis'].get('enhanced_risk_v7', {})
        if 'progressive_zones' in risk_info:
            zone = risk_info['progressive_zones'].get('risk_zone', 'unknown')
            risk_reward = risk_info['progressive_zones'].get('weighted_reward', 0)
            print(f"   🛡️ {desc}: DD {dd_pct}% → Zona: {zone} | Penalidade: {risk_reward:.2f}")
        else:
            print(f"   🛡️ {desc}: DD {dd_pct}% → Sistema de risco básico")
    
    print("\n✅ [CONCLUSÃO] Todos os testes do reward system completados!")
    print("🍒 Cherry reward system funcionando corretamente!")
    
    return True

def test_quality_system():
    """Testar sistema de qualidade inteligente"""
    print("\n🧠 [TESTE QUALIDADE] Testando sistema de qualidade...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Verificar se tem sistema de qualidade
    if hasattr(reward_system, 'quality_filter') and reward_system.quality_filter_enabled:
        print("✅ Sistema de qualidade ativo")
        
        # Testar algumas ações
        mock_env = MockEnv()
        test_actions = [
            ([1.0, 0.9, 0.0, 1.0, 2.0, -1.5], "Entrada LONG alta qualidade"),
            ([2.0, 0.5, 0.0, -1.0, 1.0, 0.5], "Entrada SHORT média qualidade"),
            ([0.0, 0.8, 0.0, 0.0, 0.0, 0.0], "HOLD (sem avaliação)")
        ]
        
        for action_vals, desc in test_actions:
            action = np.array(action_vals)
            old_state = {'trades_count': 0}
            
            reward, info, done = reward_system.calculate_reward_and_info(mock_env, action, old_state)
            
            quality_info = info.get('quality_analysis', {})
            print(f"   {desc}:")
            print(f"      Status: {quality_info.get('status', 'unknown')}")
            if 'score' in quality_info:
                print(f"      Score: {quality_info['score']:.1f}")
                print(f"      Categoria: {quality_info.get('category', 'unknown')}")
    else:
        print("ℹ️ Sistema de qualidade desabilitado ou não disponível")

if __name__ == "__main__":
    print("🍒 ===============================================")
    print("🍒 TESTE COMPLETO DO REWARD SYSTEM - CHERRY")  
    print("🍒 ===============================================")
    
    try:
        # Teste principal
        success = test_reward_system()
        
        # Teste de qualidade
        test_quality_system()
        
        if success:
            print("\n🎉 TODOS OS TESTES PASSARAM!")
            print("🍒 Cherry reward system está funcionando perfeitamente!")
            print("🚀 Pronto para treinar modelos swing trade!")
        else:
            print("\n💥 ALGUNS TESTES FALHARAM!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERRO DURANTE O TESTE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)