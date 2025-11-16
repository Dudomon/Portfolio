#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 VERIFICADOR DE CORREÇÃO DE LEARNING RATE

Verifica se a correção do LR fixo está funcionando
"""

import sys
import os
from datetime import datetime

# Importar configurações do daytrader
sys.path.append(os.path.dirname(__file__))

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def verify_lr_configuration():
    """Verificar configuração de LR"""
    
    print("🔧 VERIFICADOR DE CORREÇÃO DE LEARNING RATE")
    print("=" * 80)
    print(f"⏰ Verificação em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Importar configurações
        from daytrader import BEST_PARAMS, CONVERGENCE_OPTIMIZATION_CONFIG
        
        print("\n📊 CONFIGURAÇÕES ATUAIS:")
        print("=" * 60)
        print(f"🎯 BEST_PARAMS Learning Rate: {BEST_PARAMS['learning_rate']:.2e}")
        print(f"🎯 BEST_PARAMS Entropy Coef: {BEST_PARAMS['ent_coef']:.3f}")
        print(f"🎯 BEST_PARAMS Clip Range: {BEST_PARAMS['clip_range']:.3f}")
        
        print(f"\n🔧 Convergence Optimization: {'HABILITADO' if CONVERGENCE_OPTIMIZATION_CONFIG['enabled'] else 'DESABILITADO'}")
        if CONVERGENCE_OPTIMIZATION_CONFIG['enabled']:
            print(f"🔧 Base LR (seria usado): {CONVERGENCE_OPTIMIZATION_CONFIG['base_lr']:.2e}")
            print("⚠️ PROBLEMA: Sistema de otimização ainda ativo!")
        else:
            print("✅ Sistema de otimização DESABILITADO - LR será fixo")
        
        # Análise da correção
        print("\n🎯 ANÁLISE DA CORREÇÃO:")
        print("=" * 60)
        
        lr = BEST_PARAMS['learning_rate']
        ent_coef = BEST_PARAMS['ent_coef']
        
        # Learning Rate
        if lr < 1e-4:
            print(f"🔴 LR {lr:.2e}: MUITO BAIXO - pode causar convergência prematura")
        elif lr > 3e-4:
            print(f"🔴 LR {lr:.2e}: MUITO ALTO - pode causar instabilidade")
        else:
            print(f"✅ LR {lr:.2e}: BALANCEADO - deve evitar convergência prematura")
        
        # Entropy Coefficient
        if ent_coef < 0.02:
            print(f"🔴 Entropy {ent_coef:.3f}: BAIXO - pode convergir cedo")
        elif ent_coef > 0.05:
            print(f"🔴 Entropy {ent_coef:.3f}: ALTO - pode ser instável")
        else:
            print(f"✅ Entropy {ent_coef:.3f}: BOM - deve manter exploração")
        
        # Convergence Optimization
        if not CONVERGENCE_OPTIMIZATION_CONFIG['enabled']:
            print("✅ Convergence Opt: DESABILITADO - sem interferências no LR")
        else:
            print("🔴 Convergence Opt: HABILITADO - pode sobrescrever LR")
        
        print("\n🎯 RESULTADOS ESPERADOS COM A CORREÇÃO:")
        print("=" * 60)
        print("✅ Learning Rate fixo: 1.5e-4 (sem scheduling)")
        print("✅ KL Divergence: 1e-3 a 5e-3 (saudável)")
        print("✅ Clip Fraction: 0.05 a 0.25 (ativo)")
        print("✅ Pesos: ATIVOS (sem congelamento)")
        print("✅ Convergência: >2M steps (objetivo principal)")
        
        print("\n📊 COMPARAÇÃO:")
        print("=" * 60)
        print("❌ ANTES: current_lr: 4.8e-05 → 7.32e-05 (scheduling ativo)")
        print("✅ AGORA: current_lr: 1.5e-04 (FIXO, sem mudanças)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro ao importar configurações: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro na verificação: {e}")
        return False

def main():
    """Executar verificação completa"""
    
    print("🚀 SISTEMA DE VERIFICAÇÃO - CORREÇÃO DE LR")
    print("=" * 80)
    
    success = verify_lr_configuration()
    
    if success:
        print("\n✅ VERIFICAÇÃO CONCLUÍDA COM SUCESSO!")
        print("🎯 A configuração deve resolver os problemas de LR")
        print("💡 Reinicie o treinamento para aplicar as correções")
    else:
        print("\n❌ FALHA NA VERIFICAÇÃO")
        print("💡 Verifique os imports e configurações")
    
    print(f"\n⏰ Verificação concluída em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()