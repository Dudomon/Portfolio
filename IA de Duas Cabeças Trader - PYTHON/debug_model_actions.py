#!/usr/bin/env python3
"""
🔍 DEBUG MODEL ACTIONS - Verificar se modelo produz NaN/Inf nas ações
"""

import numpy as np
import sys
sys.path.append('.')

def check_recent_evaluation_logs():
    """Verificar logs de avaliação para actions problemáticas"""
    
    print("🔍 VERIFICANDO SE MODELO PRODUZ NaN/Inf...")
    
    # Simular ações que podem vir do modelo
    test_actions = [
        np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Normal
        np.array([1.0, 0.8, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]),  # SL NaN
        np.array([1.0, 0.8, 0.0, np.inf, 0.0, 0.0, 0.0, 0.0]),  # TP Inf
        np.array([1.0, 0.8, -np.inf, np.nan, 0.0, 0.0, 0.0, 0.0]),  # Ambos problemáticos
        np.array([np.nan] * 8),  # Todas NaN
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n📋 Teste {i+1}: Action = {action}")
        
        # Extrair SL/TP values como no daytrader.py
        try:
            sl_global = float(action[2])  # [-3,3] SL global
            tp_global = float(action[3])  # [-3,3] TP global
            sl_pos1 = float(action[4])    # [-3,3] SL específico pos 1
            tp_pos1 = float(action[5])    # [-3,3] TP específico pos 1
            
            print(f"   Extraído: SL_global={sl_global}, TP_global={tp_global}")
            print(f"            SL_pos1={sl_pos1}, TP_pos1={tp_pos1}")
            
            # Verificar valores problemáticos
            for name, value in [("SL_global", sl_global), ("TP_global", tp_global), 
                               ("SL_pos1", sl_pos1), ("TP_pos1", tp_pos1)]:
                if np.isnan(value):
                    print(f"   🚨 {name} é NaN!")
                elif np.isinf(value):
                    print(f"   🚨 {name} é Inf!")
                elif abs(value) > 10:
                    print(f"   ⚠️ {name} fora de range: {value}")
            
        except Exception as e:
            print(f"   ❌ ERRO ao extrair valores: {e}")

def simulate_position_creation_failure():
    """Simular falha na criação de posição devido a NaN/Inf"""
    
    print("\n🔍 SIMULANDO FALHA NA CRIAÇÃO DE POSIÇÃO...")
    
    # Simular dados como no daytrader.py
    current_price = 2000.0
    
    # Caso problemático: SL/TP com NaN
    problematic_adjusts = [np.nan, 0.0]  # sl_adjust = NaN
    
    try:
        # Simular conversão (vai dar erro)
        from daytrader import REALISTIC_SLTP_CONFIG
        
        sl_adjust = problematic_adjusts[0]
        tp_adjust = problematic_adjusts[1]
        
        print(f"Tentando converter: sl_adjust={sl_adjust}, tp_adjust={tp_adjust}")
        
        # Tentar fazer a conversão manualmente
        sl_points = REALISTIC_SLTP_CONFIG['sl_min_points'] + \
                    (sl_adjust + 3) * (REALISTIC_SLTP_CONFIG['sl_max_points'] - REALISTIC_SLTP_CONFIG['sl_min_points']) / 6
        
        print(f"sl_points calculado: {sl_points}")
        
    except Exception as e:
        print(f"🚨 CONVERSÃO FALHOU: {e}")
        print("   → Posição pode ser criada SEM SL válido!")

def check_policy_output_ranges():
    """Verificar se policy pode produzir valores fora de range"""
    
    print("\n🔍 VERIFICANDO RANGES DE OUTPUT DA POLICY...")
    
    # Em redes neurais, outputs podem ser problemáticos se:
    issues = [
        "1. Gradients explodem (gradient explosion) → Inf",  
        "2. Divisão por zero em normalização → NaN",
        "3. Log de números negativos → NaN",
        "4. Overflow em ativações → Inf", 
        "5. Underflow extremo → valores muito pequenos",
    ]
    
    for issue in issues:
        print(f"   {issue}")
    
    print("\n💡 POSSÍVEIS CAUSAS NO MODELO:")
    print("   - Learning rate muito alto")
    print("   - Batch normalization instável") 
    print("   - Ativações saturam (tanh → ±1)")
    print("   - Inicialização de pesos problemática")

if __name__ == "__main__":
    check_recent_evaluation_logs()
    simulate_position_creation_failure()
    check_policy_output_ranges()
    
    print(f"\n🎯 CONCLUSÃO:")
    print(f"   Se modelo produz NaN/Inf → convert_action_to_realistic_sltp FALHA")
    print(f"   → Posição criada sem SL válido → Perdas massivas possíveis")