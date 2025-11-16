#!/usr/bin/env python3
"""
CORREÇÃO EMERGENCIAL: NaN Explosion no Modelo V7
"""
import pandas as pd
import numpy as np

def analyze_dataset_for_nans():
    """Verificar se o dataset tem valores extremos causando NaN"""
    print("=== ANÁLISE DO DATASET PARA NaN ===")
    
    dataset_path = 'data/GOLD_CHALLENGING_2M_20250801_201224.csv'
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset shape: {df.shape}")
        
        # Verificar NaN/inf no dataset
        nan_count = df.isnull().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"NaN values: {nan_count}")
        print(f"Infinite values: {inf_count}")
        
        # Verificar valores extremos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\\nVALORES EXTREMOS:")
        
        for col in numeric_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                std_val = df[col].std()
                
                print(f"  {col}: min={min_val:.2f}, max={max_val:.2f}, std={std_val:.2f}")
                
                # Detectar valores extremos (>10 desvios padrão)
                if std_val > 0:
                    extreme_mask = np.abs(df[col] - df[col].mean()) > 10 * std_val
                    extreme_count = extreme_mask.sum()
                    if extreme_count > 0:
                        print(f"    ❌ {extreme_count} valores extremos (>10σ)")
        
        # Verificar volatilidade
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            vol = returns.std()
            max_return = returns.max()
            min_return = returns.min()
            
            print(f"\\nVOLATILIDADE:")
            print(f"  Volatilidade: {vol:.4f} ({vol*100:.2f}%)")
            print(f"  Max return: {max_return:.4f} ({max_return*100:.2f}%)")  
            print(f"  Min return: {min_return:.4f} ({min_return*100:.2f}%)")
            
            if vol > 0.1:  # >10%
                print(f"  ❌ VOLATILIDADE EXTREMA! Pode causar NaN no modelo")
            elif abs(max_return) > 0.2 or abs(min_return) > 0.2:  # >20%
                print(f"  ❌ RETURNS EXTREMOS! Podem causar NaN no modelo")
                
    except Exception as e:
        print(f"Erro ao analisar dataset: {e}")

def suggest_fixes():
    """Sugerir correções para NaN explosion"""
    print(f"\\n" + "="*60)
    print("CORREÇÕES PARA NaN EXPLOSION")
    print("="*60)
    
    print(f"\\nCORREÇÃO 1: HIPERPARÂMETROS AINDA MAIS CONSERVADORES")
    print(f"  learning_rate: 5e-05 → 1e-05 (5x menor)")
    print(f"  max_grad_norm: 1.0 → 0.5 (2x menor)")
    print(f"  ent_coef: 0.3 → 0.5 (maior exploração)")
    
    print(f"\\nCORREÇÃO 2: NORMALIZAÇÃO MAIS ROBUSTA")
    print(f"  - Clip extremos antes da normalização")
    print(f"  - Adicionar epsilon para evitar divisão por zero")
    print(f"  - Usar robust scaling (mediana/IQR)")
    
    print(f"\\nCORREÇÃO 3: DATASET MENOS EXTREMO")
    print(f"  - Reduzir volatilidade: 7.15% → 2.0%")
    print(f"  - Suavizar eventos extremos")
    print(f"  - Adicionar sanity checks")
    
    print(f"\\nCORREÇÃO 4: PROTEÇÕES NO MODELO")
    print(f"  - NaN checking nas camadas")
    print(f"  - Gradient clipping mais agressivo")
    print(f"  - Reinicialização de pesos se NaN")
    
    print(f"\\nRECOMENDAÇÃO IMEDIATA:")
    print(f"1. Implementar Correção 1 (hiperparâmetros)")
    print(f"2. Testar com dataset original (menos extremo)")
    print(f"3. Se funcionar, ajustar dataset gradualmente")

def create_fixed_hyperparams():
    """Criar hiperparâmetros ainda mais conservadores"""
    print(f"\\n" + "="*60)
    print("HIPERPARÂMETROS ULTRA-ULTRA-CONSERVADORES")
    print("="*60)
    
    fixed_params = {
        "learning_rate": 1.0e-05,        # 5x menor
        "n_steps": 4096,                 # 2x maior (mais estabilidade)
        "batch_size": 256,               # 2x maior (menos noise)
        "n_epochs": 2,                   # Metade (evitar overfit)
        "gamma": 0.995,                  # Mais conservador
        "gae_lambda": 0.9,               # Menos GAE
        "clip_range": 0.05,              # Muito restritivo
        "ent_coef": 0.5,                 # Máxima exploração
        "vf_coef": 0.25,                 # Menos importância value
        "max_grad_norm": 0.5,            # Clipping muito agressivo
    }
    
    print("NOVOS PARÂMETROS:")
    for key, value in fixed_params.items():
        print(f"  {key}: {value}")
    
    print(f"\\nRESULTADO ESPERADO:")
    print(f"- Aprendizado MUITO lento mas estável")
    print(f"- Zero chance de gradient explosion")
    print(f"- NaN explosion deve ser evitado")
    print(f"- Convergência em 5-10M steps (OK)")
    
    return fixed_params

def main():
    print("DIAGNÓSTICO E CORREÇÃO DE NaN EXPLOSION")
    print("="*80)
    
    # Analisar dataset
    analyze_dataset_for_nans()
    
    # Sugerir correções
    suggest_fixes()
    
    # Criar parâmetros fixos
    fixed_params = create_fixed_hyperparams()
    
    print(f"\\n" + "="*80)
    print("AÇÃO IMEDIATA RECOMENDADA")
    print("="*80)
    
    print(f"\\n1. APLICAR hiperparâmetros ultra-ultra-conservadores")
    print(f"2. TESTAR com dataset original (menos extremo)")
    print(f"3. MONITORAR para NaN explosion")
    print(f"4. Se funcionar, aumentar complexidade gradualmente")
    
    print(f"\\nO problema é que criamos um dataset MUITO desafiador")
    print(f"(volatilidade 7.15%) que está causando instabilidade numérica.")
    print(f"Precisamos de abordagem mais gradual!")

if __name__ == '__main__':
    main()