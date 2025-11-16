#!/usr/bin/env python3
"""
ANÁLISE DO DATASET SINTÉTICO ATUAL - POTENCIAL PARA ADICIONAR DIFICULDADE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_dataset(csv_path):
    """Analisar características do dataset sintético"""
    print(f"\\n=== ANÁLISE: {csv_path.name} ===")
    
    # Carregar dados
    try:
        df = pd.read_csv(csv_path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Análise básica de OHLCV
        if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            print(f"\\nOHLCV STATS:")
            print(f"  Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
            print(f"  Volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
            
            # Volatilidade
            df['returns'] = df['close'].pct_change()
            daily_vol = df['returns'].std()
            print(f"  Daily volatility: {daily_vol:.4f} ({daily_vol*100:.2f}%)")
            
            # Movimento direcional
            up_days = (df['returns'] > 0).sum()
            down_days = (df['returns'] < 0).sum()
            print(f"  Up days: {up_days} ({up_days/len(df)*100:.1f}%)")
            print(f"  Down days: {down_days} ({down_days/len(df)*100:.1f}%)")
            
            # Análise de complexidade
            print(f"\\nCOMPLEXIDADE ATUAL:")
            
            # 1. Reversões de tendência
            price_changes = np.diff(df['close'])
            direction_changes = np.sum(np.diff(np.sign(price_changes)) != 0)
            reversal_rate = direction_changes / len(price_changes)
            print(f"  Taxa de reversão: {reversal_rate:.3f} (mudanças direção por bar)")
            
            # 2. Volatilidade clustering
            vol_changes = np.abs(df['returns'].rolling(10).std().diff())
            vol_stability = 1 - vol_changes.std()
            print(f"  Estabilidade volatilidade: {vol_stability:.3f} (1=estável, 0=chaos)")
            
            # 3. Padrões repetitivos
            price_normalized = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
            autocorr_1 = price_normalized.autocorr(lag=1)
            autocorr_10 = price_normalized.autocorr(lag=10)
            print(f"  Autocorrelação lag-1: {autocorr_1:.3f}")
            print(f"  Autocorrelação lag-10: {autocorr_10:.3f}")
            
            # 4. Análise de microestrutura
            spread_proxy = (df['high'] - df['low']) / df['close']
            avg_spread = spread_proxy.mean()
            print(f"  Spread médio: {avg_spread:.4f} ({avg_spread*100:.2f}%)")
            
            # DIAGNÓSTICO DE DIFICULDADE
            print(f"\\nDIAGNÓSTICO DE DIFICULDADE:")
            
            difficulty_score = 0
            max_score = 5
            
            # Critério 1: Volatilidade baixa = fácil
            if daily_vol < 0.01:  # < 1%
                print(f"  ❌ Volatilidade BAIXA: {daily_vol*100:.2f}% (fácil demais)")
            elif daily_vol < 0.02:
                print(f"  ⚠️ Volatilidade MODERADA: {daily_vol*100:.2f}%")
                difficulty_score += 1
            else:
                print(f"  ✅ Volatilidade ALTA: {daily_vol*100:.2f}% (desafiador)")
                difficulty_score += 2
            
            # Critério 2: Reversões frequentes = difícil
            if reversal_rate < 0.3:
                print(f"  ❌ Poucas reversões: {reversal_rate:.3f} (tendências muito claras)")
            elif reversal_rate < 0.5:
                print(f"  ⚠️ Reversões moderadas: {reversal_rate:.3f}")
                difficulty_score += 1
            else:
                print(f"  ✅ Muitas reversões: {reversal_rate:.3f} (imprevisível)")
                difficulty_score += 2
            
            # Critério 3: Baixa autocorrelação = difícil
            if abs(autocorr_1) > 0.8:
                print(f"  ❌ Alta previsibilidade: autocorr={autocorr_1:.3f}")
            elif abs(autocorr_1) > 0.5:
                print(f"  ⚠️ Previsibilidade moderada: autocorr={autocorr_1:.3f}")
                difficulty_score += 1
            else:
                print(f"  ✅ Baixa previsibilidade: autocorr={autocorr_1:.3f}")
                difficulty_score += 1
            
            print(f"\\nSCORE DE DIFICULDADE: {difficulty_score}/{max_score}")
            
            if difficulty_score <= 2:
                print(f"  🟢 DATASET FÁCIL - Modelo complexo vai overfit")
            elif difficulty_score <= 3:
                print(f"  🟡 DATASET MODERADO - Pode ser adequado")
            else:
                print(f"  🔴 DATASET DIFÍCIL - Desafiador para qualquer modelo")
            
            return {
                'path': csv_path.name,
                'rows': len(df),
                'volatility': daily_vol,
                'reversal_rate': reversal_rate,
                'autocorr': autocorr_1,
                'spread': avg_spread,
                'difficulty_score': difficulty_score,
                'max_difficulty': max_score
            }
            
    except Exception as e:
        print(f"ERRO ao analisar {csv_path}: {e}")
        return None

def suggest_improvements(analysis_results):
    """Sugerir melhorias baseadas na análise"""
    print(f"\\n" + "="*80)
    print(f"SUGESTÕES PARA AUMENTAR DIFICULDADE SEM PERDER QUALIDADE")
    print(f"="*80)
    
    best_dataset = max(analysis_results, key=lambda x: x['difficulty_score'])
    worst_dataset = min(analysis_results, key=lambda x: x['difficulty_score'])
    
    print(f"\\nMELHOR DATASET ATUAL: {best_dataset['path']} (score: {best_dataset['difficulty_score']}/{best_dataset['max_difficulty']})")
    print(f"PIOR DATASET ATUAL: {worst_dataset['path']} (score: {worst_dataset['difficulty_score']}/{worst_dataset['max_difficulty']})")
    
    print(f"\\nMELHORIAS PROPOSTAS:")
    
    print(f"\\n1. ADICIONAR REGIME CHANGES:")
    print(f"   - Alternar entre bull/bear/sideways a cada 200-500k steps")
    print(f"   - Volatilidade diferente por regime (bull: 1%, bear: 3%, sideways: 0.5%)")
    print(f"   - Correlações que mudam entre regimes")
    
    print(f"\\n2. MICROESTRUTURA REALISTA:")
    print(f"   - Bid/ask spread variável (0.01% - 0.1%)")
    print(f"   - Slippage baseado em volume")
    print(f"   - Market impact (ordens grandes movem preço)")
    
    print(f"\\n3. EVENTOS EXTREMOS:")
    print(f"   - Flash crashes (5-10% em poucos minutos)")
    print(f"   - Gap ups/downs (aberturas com gap)")
    print(f"   - Períodos de baixa liquidez (spreads altos)")
    
    print(f"\\n4. CORRELAÇÕES DINÂMICAS:")
    print(f"   - Correlação com outros ativos varia no tempo")
    print(f"   - Breakdowns de correlação em crises")
    print(f"   - Lead-lag relationships")
    
    print(f"\\n5. NOISE INTELIGENTE:")
    print(f"   - Adicionar noise que não é random (patterns falsos)")
    print(f"   - Head & shoulders falsos")
    print(f"   - Breakouts falsos de suporte/resistência")
    
    print(f"\\n6. CALENDÁRIO ECONÔMICO:")
    print(f"   - Maior volatilidade em 'announcements' simulados")
    print(f"   - Padrões intraday (abertura, fechamento)")
    print(f"   - Efeitos de fim de mês/trimestre")

def main():
    print("ANÁLISE DE DIFICULDADE DO DATASET SINTÉTICO")
    print("="*80)
    
    # Analisar todos os datasets disponíveis
    data_dir = Path("data")
    csv_files = list(data_dir.glob("GOLD_SYNTHETIC*.csv"))
    
    if not csv_files:
        print("❌ Nenhum dataset sintético encontrado em data/")
        return
    
    results = []
    for csv_file in csv_files:
        result = analyze_dataset(csv_file)
        if result:
            results.append(result)
    
    if results:
        suggest_improvements(results)
        
        print(f"\\n" + "="*80)
        print(f"RECOMENDAÇÃO FINAL")
        print(f"="*80)
        
        avg_difficulty = sum(r['difficulty_score'] for r in results) / len(results)
        print(f"\\nDificuldade média atual: {avg_difficulty:.1f}/5.0")
        
        if avg_difficulty < 3.0:
            print(f"\\n🚨 DATASETS MUITO FÁCEIS PARA MODELO V7!")
            print(f"\\nAÇÃO RECOMENDADA:")
            print(f"1. USAR dataset REALISTIC (mais difícil)")
            print(f"2. IMPLEMENTAR melhorias 1, 2 e 3 acima")
            print(f"3. TESTAR com modelo V7 + hiperparâmetros conservadores")
            
        print(f"\\nIMPLEMENTAR EM:")
        print(f"- create_enhanced_gold_dataset.py")
        print(f"- Usar no daytrader.py com ULTRA_CONSERVATIVE_PARAMS")

if __name__ == '__main__':
    main()