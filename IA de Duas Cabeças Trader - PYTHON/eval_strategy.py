#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AVALIAÇÃO RÁPIDA DE ESTRATÉGIA - CRIAR ARQUIVO eval.txt

Script simples para ativar avaliação on-demand do modelo durante o treinamento.
"""

import os
import sys
from datetime import datetime

def create_eval_trigger():
    """Criar arquivo eval.txt para ativar avaliação"""
    
    eval_content = f"""EVAL_REQUEST
timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
tests_requested:
- strategy_validation
- pattern_recognition  
- risk_management
- performance_analysis
requested_by: user
"""
    
    try:
        with open('eval.txt', 'w', encoding='utf-8') as f:
            f.write(eval_content)
        
        print("OK Arquivo eval.txt criado!")
        print("Sistema de avaliacao sera ativado no proximo checkpoint")
        print("Aguarde alguns minutos para ver os resultados...")
        
        return True
        
    except Exception as e:
        print(f"ERRO ao criar eval.txt: {e}")
        return False

def run_immediate_strategy_validation():
    """Executar validação estratégica imediatamente"""
    
    print("EXECUTANDO VALIDACAO ESTRATEGICA IMEDIATA...")
    
    try:
        # Importar e executar sistema de validação
        from strategy_validation_system import StrategyValidator
        
        validator = StrategyValidator()
        
        if validator.load_model_and_env():
            # Executar testes principais
            print("\nExecutando testes de validacao...")
            
            validator.test_market_regime_adaptation()
            validator.test_technical_pattern_recognition()
            validator.test_risk_management_logic()
            validator.test_strategy_consistency()
            
            # Gerar relatório
            score = validator.generate_strategy_report()
            
            return score
        else:
            print("ERRO - Nao foi possivel carregar componentes para validacao")
            return 0
            
    except Exception as e:
        print(f"ERRO na validacao: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    """Menu principal de avaliação"""
    
    print("SISTEMA DE AVALIACAO DE ESTRATEGIAS")
    print("=" * 50)
    print("1. Criar eval.txt (avaliação durante treinamento)")
    print("2. Executar validação imediata")
    print("3. Ambos")
    
    try:
        choice = input("\nEscolha uma opção (1-3): ").strip()
        
        if choice == "1":
            create_eval_trigger()
        elif choice == "2":
            score = run_immediate_strategy_validation()
            print(f"\nScore final de validacao: {score:.1%}")
        elif choice == "3":
            create_eval_trigger()
            print("\n" + "="*50)
            score = run_immediate_strategy_validation()
            print(f"\nScore final de validacao: {score:.1%}")
        else:
            print("Opcao invalida")
            
    except KeyboardInterrupt:
        print("\n\nAvaliacao cancelada pelo usuario")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()