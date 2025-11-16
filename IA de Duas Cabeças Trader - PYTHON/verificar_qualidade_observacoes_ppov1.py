#!/usr/bin/env python3
"""
🔍 VERIFICAR QUALIDADE DAS OBSERVAÇÕES - PPOV1.PY
Analisa a qualidade das observações recebidas pelo modelo
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Adicionar paths
sys.path.append(".")

def verificar_qualidade_observacoes():
    """Verificar qualidade das observações no ppov1.py"""
    
    print("🔍 VERIFICANDO QUALIDADE DAS OBSERVAÇÕES - PPOV1.PY")
    print("=" * 70)
    
    try:
        # Importar ppov1
        from ppov1 import TradingEnv, load_optimized_data
        
        # 1. CARREGAR DADOS
        print("\n1. 📊 CARREGANDO DADOS")
        print("-" * 40)
        
        df = load_optimized_data()
        print(f"✅ Dataset carregado: {len(df):,} barras")
        print(f"📅 Período: {df.index[0]} → {df.index[-1]}")
        
        # 2. CRIAR AMBIENTE
        print("\n2. 🧪 CRIANDO AMBIENTE DE TESTE")
        print("-" * 40)
        
        env = TradingEnv(df, window_size=20, is_training=True, initial_balance=500)
        print(f"✅ Ambiente criado")
        print(f"📊 Observation space: {env.observation_space.shape}")
        print(f"🎯 Action space: {env.action_space.shape}")
        
        # 3. TESTAR MÚLTIPLAS OBSERVAÇÕES
        print("\n3. 🔍 TESTANDO QUALIDADE DAS OBSERVAÇÕES")
        print("-" * 40)
        
        n_tests = 100
        observacoes_analisadas = []
        
        for i in range(n_tests):
            # Reset ambiente
            obs = env.reset()
            
            # Verificar observação
            if obs is not None and len(obs) > 0:
                observacoes_analisadas.append(obs)
                
                if i < 5:  # Mostrar detalhes das primeiras 5
                    print(f"\n📊 Observação {i+1}:")
                    print(f"   Shape: {obs.shape}")
                    print(f"   Tipo: {type(obs)}")
                    print(f"   Range: [{obs.min():.6f}, {obs.max():.6f}]")
                    print(f"   Média: {obs.mean():.6f}")
                    print(f"   Std: {obs.std():.6f}")
                    
                    # Verificar componentes
                    total_features = len(obs)
                    market_features = 20 * 27  # 20 steps × 27 features
                    position_features = 20 * 27  # 20 steps × 27 features  
                    intelligent_features = 20 * 12  # 20 steps × 12 features
                    
                    print(f"   Componentes:")
                    print(f"     Market: {market_features} (steps 0-539)")
                    print(f"     Position: {position_features} (steps 540-1079)")
                    print(f"     Intelligent: {intelligent_features} (steps 1080-1319)")
                    
                    # Verificar se há NaN ou Inf
                    nan_count = np.isnan(obs).sum()
                    inf_count = np.isinf(obs).sum()
                    zero_count = (obs == 0).sum()
                    
                    print(f"   Qualidade:")
                    print(f"     NaN: {nan_count}")
                    print(f"     Inf: {inf_count}")
                    print(f"     Zeros: {zero_count} ({zero_count/total_features*100:.1f}%)")
        
        # 4. ANÁLISE ESTATÍSTICA GERAL
        print(f"\n4. 📈 ANÁLISE ESTATÍSTICA GERAL ({len(observacoes_analisadas)} observações)")
        print("-" * 40)
        
        if observacoes_analisadas:
            obs_array = np.array(observacoes_analisadas)
            
            print(f"📊 Estatísticas Gerais:")
            print(f"   Média global: {obs_array.mean():.6f}")
            print(f"   Std global: {obs_array.std():.6f}")
            print(f"   Min global: {obs_array.min():.6f}")
            print(f"   Max global: {obs_array.max():.6f}")
            
            # Verificar consistência
            shapes = [obs.shape for obs in observacoes_analisadas]
            unique_shapes = set(shapes)
            print(f"   Shapes únicos: {unique_shapes}")
            
            # Verificar qualidade
            total_nan = sum(np.isnan(obs).sum() for obs in observacoes_analisadas)
            total_inf = sum(np.isinf(obs).sum() for obs in observacoes_analisadas)
            total_zeros = sum((obs == 0).sum() for obs in observacoes_analisadas)
            total_elements = sum(len(obs) for obs in observacoes_analisadas)
            
            print(f"\n🔍 Qualidade das Observações:")
            print(f"   Total elementos: {total_elements:,}")
            print(f"   NaN: {total_nan} ({total_nan/total_elements*100:.3f}%)")
            print(f"   Inf: {total_inf} ({total_inf/total_elements*100:.3f}%)")
            print(f"   Zeros: {total_zeros:,} ({total_zeros/total_elements*100:.1f}%)")
            
            # 5. ANÁLISE POR COMPONENTES
            print(f"\n5. 🧠 ANÁLISE POR COMPONENTES")
            print("-" * 40)
            
            # Market features (0-539)
            market_obs = obs_array[:, :540]
            print(f"📊 Market Features (0-539):")
            print(f"   Média: {market_obs.mean():.6f}")
            print(f"   Std: {market_obs.std():.6f}")
            print(f"   Range: [{market_obs.min():.6f}, {market_obs.max():.6f}]")
            print(f"   Zeros: {(market_obs == 0).sum()/market_obs.size*100:.1f}%")
            
            # Position features (540-1079)
            position_obs = obs_array[:, 540:1080]
            print(f"🎯 Position Features (540-1079):")
            print(f"   Média: {position_obs.mean():.6f}")
            print(f"   Std: {position_obs.std():.6f}")
            print(f"   Range: [{position_obs.min():.6f}, {position_obs.max():.6f}]")
            print(f"   Zeros: {(position_obs == 0).sum()/position_obs.size*100:.1f}%")
            
            # Intelligent features (1080-1319)
            intelligent_obs = obs_array[:, 1080:1320]
            print(f"🧠 Intelligent Features (1080-1319):")
            print(f"   Média: {intelligent_obs.mean():.6f}")
            print(f"   Std: {intelligent_obs.std():.6f}")
            print(f"   Range: [{intelligent_obs.min():.6f}, {intelligent_obs.max():.6f}]")
            print(f"   Zeros: {(intelligent_obs == 0).sum()/intelligent_obs.size*100:.1f}%")
            
            # 6. VERIFICAR NORMALIZAÇÃO
            print(f"\n6. 📏 VERIFICANDO NORMALIZAÇÃO")
            print("-" * 40)
            
            # Verificar se os dados estão normalizados
            market_std = market_obs.std()
            position_std = position_obs.std()
            intelligent_std = intelligent_obs.std()
            
            print(f"📊 Desvio Padrão por Componente:")
            print(f"   Market: {market_std:.6f}")
            print(f"   Position: {position_std:.6f}")
            print(f"   Intelligent: {intelligent_std:.6f}")
            
            if market_std < 0.1 and position_std < 0.1 and intelligent_std < 0.1:
                print("⚠️  AVISO: Desvios padrão muito baixos - possível over-normalização")
            elif market_std > 10 or position_std > 10 or intelligent_std > 10:
                print("⚠️  AVISO: Desvios padrão muito altos - possível falta de normalização")
            else:
                print("✅ Normalização parece adequada")
            
            # 7. VERIFICAR CORRELAÇÕES
            print(f"\n7. 🔗 VERIFICANDO CORRELAÇÕES")
            print("-" * 40)
            
            # Correlação entre componentes (usar amostra para evitar erro de dimensões)
            sample_size = min(10000, market_obs.size, position_obs.size, intelligent_obs.size)
            
            market_sample = market_obs.flatten()[:sample_size]
            position_sample = position_obs.flatten()[:sample_size]
            intelligent_sample = intelligent_obs.flatten()[:sample_size]
            
            corr_market_position = np.corrcoef(market_sample, position_sample)[0,1]
            corr_market_intelligent = np.corrcoef(market_sample, intelligent_sample)[0,1]
            corr_position_intelligent = np.corrcoef(position_sample, intelligent_sample)[0,1]
            
            print(f"📊 Correlações entre Componentes:")
            print(f"   Market ↔ Position: {corr_market_position:.3f}")
            print(f"   Market ↔ Intelligent: {corr_market_intelligent:.3f}")
            print(f"   Position ↔ Intelligent: {corr_position_intelligent:.3f}")
            
            if abs(corr_market_position) > 0.8:
                print("⚠️  AVISO: Alta correlação entre Market e Position")
            if abs(corr_market_intelligent) > 0.8:
                print("⚠️  AVISO: Alta correlação entre Market e Intelligent")
            if abs(corr_position_intelligent) > 0.8:
                print("⚠️  AVISO: Alta correlação entre Position e Intelligent")
            
            # 8. CONCLUSÃO
            print(f"\n8. 🎯 CONCLUSÃO DA QUALIDADE")
            print("-" * 40)
            
            qualidade_score = 100
            
            # Penalizar por NaN/Inf
            if total_nan > 0:
                qualidade_score -= 20
                print("❌ Penalidade: NaN encontrados")
            if total_inf > 0:
                qualidade_score -= 20
                print("❌ Penalidade: Inf encontrados")
            
            # Penalizar por muitos zeros
            zero_percent = total_zeros/total_elements*100
            if zero_percent > 50:
                qualidade_score -= 30
                print(f"❌ Penalidade: Muitos zeros ({zero_percent:.1f}%)")
            elif zero_percent > 30:
                qualidade_score -= 15
                print(f"⚠️  Penalidade: Zeros moderados ({zero_percent:.1f}%)")
            
            # Penalizar por correlações altas
            if abs(corr_market_position) > 0.8 or abs(corr_market_intelligent) > 0.8 or abs(corr_position_intelligent) > 0.8:
                qualidade_score -= 10
                print("❌ Penalidade: Alta correlação entre componentes")
            
            # Penalizar por normalização inadequada
            if market_std < 0.1 or position_std < 0.1 or intelligent_std < 0.1:
                qualidade_score -= 15
                print("❌ Penalidade: Over-normalização")
            elif market_std > 10 or position_std > 10 or intelligent_std > 10:
                qualidade_score -= 15
                print("❌ Penalidade: Falta de normalização")
            
            print(f"\n🏆 SCORE DE QUALIDADE: {qualidade_score}/100")
            
            if qualidade_score >= 90:
                print("✅ EXCELENTE: Observações de alta qualidade")
            elif qualidade_score >= 70:
                print("✅ BOM: Observações adequadas")
            elif qualidade_score >= 50:
                print("⚠️  MODERADO: Observações com problemas menores")
            else:
                print("❌ PROBLEMÁTICO: Observações com problemas sérios")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verificar_qualidade_observacoes() 