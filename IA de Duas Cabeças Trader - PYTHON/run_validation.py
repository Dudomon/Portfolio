#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXECUTAR VALIDACAO DIRETA - SEM MENU INTERATIVO
"""

import sys
import os

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def run_validation():
    """Executar validação diretamente"""
    print("EXECUTANDO VALIDACAO ESTRATEGICA...")
    
    try:
        from strategy_validation_system import StrategyValidator
        
        validator = StrategyValidator()
        
        if validator.load_model_and_env():
            print("\nExecutando testes de validacao...")
            
            # Executar todos os testes
            validator.test_market_regime_adaptation()
            validator.test_technical_pattern_recognition()
            validator.test_risk_management_logic()
            validator.test_strategy_consistency()
            
            # Gerar relatório
            score = validator.generate_strategy_report()
            
            print(f"\nSCORE FINAL DE VALIDACAO: {score:.1%}")
            
            if score > 0.7:
                print("RESULTADO: ESTRATEGIA VALIDA - Modelo aprendeu logica solida")
            elif score > 0.5:
                print("RESULTADO: ESTRATEGIA QUESTIONAVEL - Precisa mais treinamento")
            else:
                print("RESULTADO: ESTRATEGIA INVALIDA - Nao aprendeu logica valida")
            
            return score
        else:
            print("ERRO - Nao foi possivel carregar componentes")
            return 0
            
    except Exception as e:
        print(f"ERRO: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    run_validation()