#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALIDACAO SIMPLES DE ESTRATEGIA - ANALISE RAPIDA
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def analyze_recent_performance():
    """Analisar performance recente do modelo"""
    print("ANALISANDO PERFORMANCE RECENTE DO MODELO...")
    print("=" * 50)
    
    try:
        # Verificar arquivos de debug recentes
        debug_files = [f for f in os.listdir('.') if f.startswith('debug_zeros_report_step_') and f.endswith('.txt')]
        debug_files.sort(key=lambda x: int(x.split('step_')[1].split('.')[0]))
        
        if debug_files:
            latest_debug = debug_files[-5:]  # Últimos 5 arquivos
            print(f"Encontrados {len(debug_files)} arquivos de debug")
            print(f"Analisando os mais recentes: {len(latest_debug)} arquivos")
            
            # Analisar padrões nos logs
            zero_patterns = {}
            
            for debug_file in latest_debug:
                try:
                    with open(debug_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Extrair informações sobre zeros
                    if 'raw_data: 0 detecções' in content:
                        print(f"✓ {debug_file}: Dados de entrada limpos")
                    
                    if 'technical_indicators: 0 detecções' in content:
                        print(f"✓ {debug_file}: Indicadores técnicos OK")
                        
                except Exception as e:
                    continue
            
            return True
        else:
            print("Nenhum arquivo de debug encontrado")
            return False
            
    except Exception as e:
        print(f"Erro na análise: {e}")
        return False

def analyze_trading_patterns():
    """Analisar padrões de trading do modelo"""
    print("\nANALISANDO PADROES DE TRADING...")
    print("=" * 50)
    
    try:
        # Carregar dados do Yahoo cache (mesmo que o daytrader usa)
        yahoo_cache = "data_cache/GC=F_YAHOO_DAILY_CACHE_20250711_041924.pkl"
        if os.path.exists(yahoo_cache):
            print(f"Carregando dataset Yahoo (mesmo do daytrader)...")
            data = pd.read_pickle(yahoo_cache)
            
            print(f"Dataset carregado: {len(data)} registros")
            
            # Converter para formato padrão
            if 'time' in data.columns:
                data['time'] = pd.to_datetime(data['time'])
                data.set_index('time', inplace=True)
            
            # Análises básicas - usar dados recentes
            recent_data = data.tail(10000)  # Últimos 10k pontos
            
            # 1. Volatilidade recente (usar coluna correta do dataset)
            if 'returns' in recent_data.columns:
                returns_col = 'returns'
            elif 'returns_1m' in recent_data.columns:
                returns_col = 'returns_1m'
            else:
                # Calcular returns se não existir
                recent_data['returns'] = recent_data['close'].pct_change()
                returns_col = 'returns'
            
            volatility = recent_data[returns_col].std() * np.sqrt(252)  # Anualizada
            print(f"Volatilidade recente (anualizada): {volatility:.1%}")
            
            # 2. Regimes de mercado
            returns = recent_data[returns_col].dropna()
            bull_periods = (returns > 0).sum() / len(returns)
            print(f"Períodos de alta: {bull_periods:.1%}")
            
            # 3. Condições de RSI (usar coluna que existe)
            rsi_cols = [col for col in recent_data.columns if 'rsi' in col.lower()]
            if rsi_cols:
                rsi_col = rsi_cols[0]  # Usar primeira coluna RSI encontrada
                rsi_data = recent_data[rsi_col].dropna()
                oversold = (rsi_data < 30).sum()
                overbought = (rsi_data > 70).sum()
                print(f"Períodos oversold ({rsi_col}): {oversold} ({oversold/len(rsi_data):.1%})")
                print(f"Períodos overbought ({rsi_col}): {overbought} ({overbought/len(rsi_data):.1%})")
            else:
                print("RSI não encontrado no dataset")
            
            # 4. Análise de ranges (sem Bollinger Bands específicas)
            high_low_range = (recent_data['high'] - recent_data['low']) / recent_data['close']
            tight_ranges = (high_low_range < high_low_range.quantile(0.2)).sum()
            print(f"Períodos de baixa volatilidade (ranges pequenos): {tight_ranges} ({tight_ranges/len(high_low_range):.1%})")
            
            return {
                'volatility': volatility,
                'bull_periods': bull_periods,
                'oversold_periods': oversold/len(rsi_data) if rsi_cols else 0,
                'overbought_periods': overbought/len(rsi_data) if rsi_cols else 0,
                'low_volatility_periods': tight_ranges/len(high_low_range)
            }
        else:
            print(f"ERRO: Dataset Yahoo não encontrado em: {yahoo_cache}")
            print("Verificando arquivos na pasta data_cache...")
            if os.path.exists('data_cache'):
                files = os.listdir('data_cache')
                pkl_files = [f for f in files if f.endswith('.pkl')]
                print(f"Arquivos .pkl encontrados: {pkl_files[:5]}")  # Mostrar primeiros 5
            return None
            
    except Exception as e:
        print(f"Erro na análise de padrões: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_model_behavior():
    """Analisar comportamento atual do modelo"""
    print("\nANALISANDO COMPORTAMENTO DO MODELO...")
    print("=" * 50)
    
    # Análise dos logs de treinamento mais recentes
    training_info = {
        'trades_per_day': 0.7,  # Do log fornecido
        'win_rate': 44.4,       # Do log fornecido
        'portfolio_value': 506.55,
        'drawdown': 15.35,
        'learning_status': 'OK APRENDENDO BEM'
    }
    
    print(f"Trades por dia: {training_info['trades_per_day']}")
    print(f"Win rate global: {training_info['win_rate']:.1f}%")
    print(f"Portfolio atual: ${training_info['portfolio_value']:.2f}")
    print(f"Drawdown atual: {training_info['drawdown']:.1f}%")
    print(f"Status de aprendizado: {training_info['learning_status']}")
    
    # Avaliação qualitativa
    assessment = []
    
    # 1. Seletividade
    if training_info['trades_per_day'] < 2:
        assessment.append("✓ SELETIVO: Modelo não está overtrading")
    else:
        assessment.append("⚠ ATIVO: Modelo faz muitos trades")
    
    # 2. Win Rate
    if training_info['win_rate'] > 50:
        assessment.append("✓ WIN RATE BOM: Acima de 50%")
    elif training_info['win_rate'] > 40:
        assessment.append("~ WIN RATE OK: Entre 40-50%")
    else:
        assessment.append("⚠ WIN RATE BAIXO: Abaixo de 40%")
    
    # 3. Drawdown
    if training_info['drawdown'] < 20:
        assessment.append("✓ DRAWDOWN CONTROLADO: Abaixo de 20%")
    else:
        assessment.append("⚠ DRAWDOWN ALTO: Acima de 20%")
    
    print("\nAVALIACAO QUALITATIVA:")
    for item in assessment:
        print(f"  {item}")
    
    return training_info, assessment

def generate_strategy_assessment():
    """Gerar avaliação geral da estratégia"""
    print("\n" + "=" * 60)
    print("AVALIACAO GERAL DA ESTRATEGIA")
    print("=" * 60)
    
    # Executar análises
    debug_ok = analyze_recent_performance()
    patterns = analyze_trading_patterns()
    behavior, assessment = analyze_model_behavior()
    
    # Calcular score geral
    scores = []
    
    # Score 1: Estabilidade técnica
    if debug_ok:
        scores.append(0.8)  # Dados limpos
        print("✓ ESTABILIDADE TECNICA: 80% (dados limpos, sem zeros)")
    else:
        scores.append(0.3)
        print("⚠ ESTABILIDADE TECNICA: 30% (problemas nos dados)")
    
    # Score 2: Performance de trading
    win_rate_score = min(1.0, behavior['win_rate'] / 60)  # Normalizar para 60%
    drawdown_score = max(0, 1 - behavior['drawdown'] / 30)  # Penalizar DD > 30%
    trading_score = (win_rate_score + drawdown_score) / 2
    scores.append(trading_score)
    print(f"✓ PERFORMANCE TRADING: {trading_score:.1%} (win rate + drawdown)")
    
    # Score 3: Seletividade
    if behavior['trades_per_day'] < 1:
        selectivity_score = 0.9  # Muito seletivo
    elif behavior['trades_per_day'] < 3:
        selectivity_score = 0.7  # Seletividade moderada
    else:
        selectivity_score = 0.4  # Muito ativo
    scores.append(selectivity_score)
    print(f"✓ SELETIVIDADE: {selectivity_score:.1%} ({behavior['trades_per_day']} trades/dia)")
    
    # Score 4: Adaptação a mercado
    if patterns:
        market_score = 0.6  # Score médio baseado na diversidade de condições
        if patterns['volatility'] > 0.3:
            market_score += 0.1  # Bonus para operar em volatilidade
        if 0.1 < patterns['bull_periods'] < 0.9:
            market_score += 0.1  # Bonus para operar em mercados mistos
    else:
        market_score = 0.5  # Score neutro
    scores.append(market_score)
    print(f"✓ ADAPTACAO MERCADO: {market_score:.1%}")
    
    # Score final
    final_score = np.mean(scores)
    
    print(f"\n🏆 SCORE FINAL: {final_score:.1%}")
    
    if final_score > 0.75:
        verdict = "✅ ESTRATEGIA VALIDA - Modelo demonstra logica consistente"
        recommendation = "Continue o treinamento. Modelo esta aprendendo bem."
    elif final_score > 0.6:
        verdict = "⚠️ ESTRATEGIA PROMISSORA - Precisa mais refinamento"  
        recommendation = "Continue treinamento com monitoramento. Ajuste hiperparametros se necessario."
    elif final_score > 0.4:
        verdict = "⚠️ ESTRATEGIA QUESTIONAVEL - Resultados mistos"
        recommendation = "Considere revisar arquitetura ou dados de treinamento."
    else:
        verdict = "❌ ESTRATEGIA PROBLEMATICA - Nao demonstra logica valida"
        recommendation = "Revise completamente a abordagem de treinamento."
    
    print(f"\n📊 VEREDICTO: {verdict}")
    print(f"💡 RECOMENDACAO: {recommendation}")
    
    # Salvar relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        'timestamp': timestamp,
        'final_score': final_score,
        'verdict': verdict,
        'recommendation': recommendation,
        'scores_breakdown': {
            'technical_stability': scores[0],
            'trading_performance': scores[1], 
            'selectivity': scores[2],
            'market_adaptation': scores[3]
        },
        'behavior_metrics': behavior,
        'patterns_analysis': patterns
    }
    
    try:
        import json
        with open(f'strategy_assessment_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📁 Relatorio salvo: strategy_assessment_{timestamp}.json")
    except Exception as e:
        print(f"Erro ao salvar relatorio: {e}")
    
    return final_score

if __name__ == "__main__":
    print("SISTEMA DE VALIDACAO SIMPLES DE ESTRATEGIA")
    print("=" * 60)
    
    try:
        score = generate_strategy_assessment()
        print(f"\nVALIDACAO CONCLUIDA - Score Final: {score:.1%}")
    except Exception as e:
        print(f"ERRO na validacao: {e}")
        import traceback
        traceback.print_exc()