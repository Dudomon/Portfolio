#!/usr/bin/env python3
"""
🔍 DEBUG INDIVIDUAL SCORES V7 - Verificar saídas das redes individuais
"""

import sys
import os
sys.path.append("D:/Projeto")

import torch
import numpy as np
from sb3_contrib import RecurrentPPO

# ========== CONFIGURAÇÃO ==========
CHECKPOINT_NAME = "checkpoint_7700000_steps_20250808_165028.zip"
CHECKPOINT_PATH = f"D:/Projeto/trading_framework/training/checkpoints/DAYTRADER/{CHECKPOINT_NAME}"
# ==================================

def debug_individual_scores():
    """🔍 Analisar saídas das redes individuais que geram os scores"""
    
    print("🔍 DEBUG INDIVIDUAL SCORES V7")
    print("=" * 60)
    
    try:
        # Carregar modelo
        print(f"🤖 Carregando modelo...")
        model = RecurrentPPO.load(CHECKPOINT_PATH, device='cuda')
        model.policy.eval()
        
        # Gerar observação sintética
        print("📊 Gerando observações de teste...")
        batch_size = 100
        obs_dim = 2580
        
        # Observações com diferentes características
        obs_tests = {
            'zeros': torch.zeros(batch_size, obs_dim, device='cuda'),
            'normal': torch.randn(batch_size, obs_dim, device='cuda') * 0.5,
            'positive': torch.abs(torch.randn(batch_size, obs_dim, device='cuda')) * 0.3,
            'negative': -torch.abs(torch.randn(batch_size, obs_dim, device='cuda')) * 0.3,
            'extreme_pos': torch.ones(batch_size, obs_dim, device='cuda') * 2.0,
            'extreme_neg': torch.ones(batch_size, obs_dim, device='cuda') * -2.0
        }
        
        print("\n🧪 TESTANDO DIFERENTES INPUTS...")
        
        for test_name, obs in obs_tests.items():
            print(f"\n📊 TESTE: {test_name.upper()}")
            print("-" * 40)
            
            with torch.no_grad():
                # Passar pela policy para extrair features
                features = model.policy.extract_features(obs)
                
                # Acessar o entry_head
                entry_head = model.policy.entry_head
                
                # Simular o input que vai para as redes individuais
                # (seria combined_input no forward original)
                dummy_management = torch.zeros_like(features)
                dummy_market_context = torch.zeros_like(features)
                combined_input = torch.cat([features, dummy_management, dummy_market_context], dim=-1)
                
                # Testar cada rede individual
                individual_scores = {}
                
                # Temporal
                temporal_raw = entry_head.horizon_analyzer(combined_input)
                individual_scores['temporal'] = temporal_raw
                
                # MTF + Pattern
                mtf_raw = entry_head.mtf_validator(combined_input)
                pattern_raw = entry_head.pattern_memory_validator(combined_input)
                individual_scores['mtf'] = mtf_raw
                individual_scores['pattern'] = pattern_raw
                
                # Risk + Regime  
                risk_raw = entry_head.risk_gate_entry(combined_input)
                regime_raw = entry_head.regime_gate(combined_input)
                individual_scores['risk'] = risk_raw
                individual_scores['regime'] = regime_raw
                
                # Market
                lookahead_raw = entry_head.lookahead_gate(combined_input)
                fatigue_raw = entry_head.fatigue_detector(combined_input)
                individual_scores['lookahead'] = lookahead_raw
                individual_scores['fatigue'] = fatigue_raw
                
                # Quality (4 filtros)
                momentum_raw = entry_head.momentum_filter(combined_input)
                volatility_raw = entry_head.volatility_filter(combined_input)
                volume_raw = entry_head.volume_filter(combined_input)
                trend_raw = entry_head.trend_strength_filter(combined_input)
                individual_scores['momentum'] = momentum_raw
                individual_scores['volatility'] = volatility_raw
                individual_scores['volume'] = volume_raw
                individual_scores['trend'] = trend_raw
                
                # Confidence
                confidence_raw = entry_head.confidence_estimator(combined_input)
                individual_scores['confidence'] = confidence_raw
                
                # Análise estatística
                print("  Score → Min/Mean/Max/Std → Zeros%")
                for name, scores in individual_scores.items():
                    scores_cpu = scores.cpu().numpy().flatten()
                    
                    min_val = np.min(scores_cpu)
                    mean_val = np.mean(scores_cpu)
                    max_val = np.max(scores_cpu)
                    std_val = np.std(scores_cpu)
                    zero_pct = np.mean(scores_cpu < 1e-6) * 100
                    
                    # Status baseado na distribuição
                    if mean_val < 0.1:
                        status = "🔴 MUITO BAIXO"
                    elif mean_val > 0.9:
                        status = "🔥 SATURADO ALTO"
                    elif std_val < 0.05:
                        status = "❄️ SEM VARIÂNCIA"
                    else:
                        status = "✅ OK"
                    
                    print(f"  {name:12} → {min_val:.3f}/{mean_val:.3f}/{max_val:.3f}/{std_val:.3f} → {zero_pct:5.1f}% {status}")
        
        print("\n" + "=" * 60)
        print("💡 ANÁLISE FINAL")
        print("=" * 60)
        
        # Teste final: Input extremo positivo para verificar saturação
        extreme_input = torch.ones(1, obs_dim, device='cuda') * 10.0
        
        with torch.no_grad():
            features = model.policy.extract_features(extreme_input)
            entry_head = model.policy.entry_head
            
            dummy_management = torch.zeros_like(features)
            dummy_market_context = torch.zeros_like(features)
            combined_input = torch.cat([features, dummy_management, dummy_market_context], dim=-1)
            
            # Testar uma rede específica
            temporal_score = entry_head.horizon_analyzer(combined_input)
            print(f"🧪 TESTE EXTREMO (input=10.0):")
            print(f"  Temporal Score: {temporal_score.item():.6f}")
            
            if temporal_score.item() > 0.99:
                print("  🔥 CONFIRMADO: Sigmoid saturada para ALTO")
            elif temporal_score.item() < 0.01:
                print("  ❄️ CONFIRMADO: Sigmoid saturada para BAIXO")
            else:
                print("  ✅ Sigmoid funcionando normalmente")
        
        print("\n📋 DIAGNÓSTICOS POSSÍVEIS:")
        print("  • Se scores sempre < 0.1: Sigmoid saturada para baixo")
        print("  • Se scores sempre > 0.9: Sigmoid saturada para alto") 
        print("  • Se std < 0.05: Redes não aprenderam diferenciação")
        print("  • Se zeros% alto: Gradientes mortos")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_individual_scores()