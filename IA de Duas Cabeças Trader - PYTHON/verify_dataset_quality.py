import pandas as pd
import numpy as np

def verify_dataset_quality():
    print("🔍 VERIFICAÇÃO COMPLETA DE QUALIDADE DO DATASET")
    print("=" * 60)
    
    # Carregar dataset
    df = pd.read_csv('data/GOLD_TRADING_READY_2M_ENHANCED_INDICATORS.csv')
    
    # Features originais vs indicadores
    original_features = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'regime']
    indicadores = [col for col in df.columns if col not in original_features]
    
    print(f"📊 ESTRUTURA DO DATASET:")
    print(f"   • Total de linhas: {len(df):,}")
    print(f"   • Total de colunas: {len(df.columns)}")
    print(f"   • Features originais: {len(original_features)}")
    print(f"   • Indicadores técnicos: {len(indicadores)}")
    
    print(f"\n🔧 INDICADORES TÉCNICOS ADICIONADOS ({len(indicadores)}):")
    for i, ind in enumerate(indicadores, 1):
        print(f"   {i:2d}. {ind}")
    
    print(f"\n📊 QUALIDADE DOS DADOS:")
    print(f"   • Valores NaN: {df.isnull().sum().sum()}")
    print(f"   • Valores infinitos: {df.isin([float('inf'), -float('inf')]).sum().sum()}")
    print(f"   • Duplicatas: {df.duplicated().sum()}")
    
    print(f"\n📈 SAMPLE DE RANGES (primeiros 5 indicadores):")
    for ind in indicadores[:5]:
        min_val = df[ind].min()
        max_val = df[ind].max()
        mean_val = df[ind].mean()
        print(f"   {ind}: [{min_val:.4f}, {max_val:.4f}] (média: {mean_val:.4f})")
    
    print(f"\n🎯 REGIME DISTRIBUTION:")
    regime_counts = df['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {regime}: {count:,} ({pct:.1f}%)")
    
    # Verificação de continuidade temporal
    print(f"\n⏰ CONTINUIDADE TEMPORAL:")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_diffs = df['timestamp'].diff().dropna()
    print(f"   • Intervalo predominante: {time_diffs.mode()[0]}")
    print(f"   • Gaps temporais: {(time_diffs != time_diffs.mode()[0]).sum()}")
    
    # Verificação de correlações extremas
    print(f"\n🔗 CORRELAÇÕES ALTAS (>0.95):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    high_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.95:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr.append((col1, col2, corr_val))
    
    if high_corr:
        for col1, col2, corr_val in high_corr[:5]:  # Top 5
            print(f"   {col1} ↔ {col2}: {corr_val:.3f}")
    else:
        print("   ✅ Nenhuma correlação extrema detectada")
    
    print(f"\n✅ DATASET QUALITY SCORE:")
    quality_score = 100
    
    # Deduções por problemas
    if df.isnull().sum().sum() > 0:
        quality_score -= 20
        print(f"   -20: Valores NaN presentes")
    
    if df.isin([float('inf'), -float('inf')]).sum().sum() > 0:
        quality_score -= 30
        print(f"   -30: Valores infinitos presentes")
    
    if df.duplicated().sum() > 100:
        quality_score -= 10
        print(f"   -10: Muitas duplicatas")
    
    if len(high_corr) > 10:
        quality_score -= 15
        print(f"   -15: Muitas correlações extremas")
    
    if len(indicadores) < 25:
        quality_score -= 25
        print(f"   -25: Poucos indicadores técnicos")
    
    print(f"\n🏆 QUALITY SCORE: {quality_score}/100")
    
    if quality_score >= 90:
        print("✅ DATASET 100% PRONTO PARA TREINAMENTO!")
        return True
    elif quality_score >= 80:
        print("⚠️ Dataset bom, pequenos ajustes recomendados")
        return True
    else:
        print("❌ Dataset precisa de correções antes do treinamento")
        return False

if __name__ == "__main__":
    success = verify_dataset_quality()
    exit(0 if success else 1)