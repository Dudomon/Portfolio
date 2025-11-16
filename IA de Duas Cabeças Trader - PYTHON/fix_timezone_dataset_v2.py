#!/usr/bin/env python3
"""
🔧 FIX TIMEZONE DATASET V2
Corrigir problema de timezone no dataset híbrido V2
"""

import pandas as pd
from datetime import datetime
import logging

def fix_timezone_issues():
    """Corrigir problemas de timezone no dataset V2"""
    print("🔧 CORRIGINDO TIMEZONE DATASET V2")
    print("="*50)
    
    dataset_path = "data/GC=F_HYBRID_V2_3Y_1MIN_20250911_200306.csv"
    
    print(f"📂 Carregando: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print(f"✅ Dataset carregado: {len(df):,} linhas")
    print(f"🔍 Colunas: {list(df.columns)}")
    
    # Verificar coluna time
    print(f"\n📅 ANALISANDO COLUNA TIME:")
    print(f"Tipo atual: {df['time'].dtype}")
    print(f"Primeiros valores:")
    for i in range(min(5, len(df))):
        print(f"  {i}: {df['time'].iloc[i]} (tipo: {type(df['time'].iloc[i])})")
    
    # Converter time para datetime sem timezone
    print(f"\n🔧 CONVERTENDO TIME PARA UTC SEM TIMEZONE:")
    
    try:
        # Tentar conversão com UTC
        df['time'] = pd.to_datetime(df['time'], utc=True)
        print("✅ Conversão UTC bem-sucedida")
        
        # Remover timezone (converter para naive datetime)
        df['time'] = df['time'].dt.tz_localize(None)
        print("✅ Timezone removido - agora naive datetime")
        
    except Exception as e:
        print(f"❌ Erro na conversão UTC: {e}")
        
        # Fallback: força conversão string
        try:
            df['time'] = pd.to_datetime(df['time'].astype(str), errors='coerce')
            print("✅ Conversão forçada como string bem-sucedida")
        except Exception as e2:
            print(f"❌ Erro na conversão forçada: {e2}")
            return False
    
    # Verificar resultado
    print(f"\n📊 RESULTADO:")
    print(f"Tipo final: {df['time'].dtype}")
    print(f"Timezone: {df['time'].dt.tz if hasattr(df['time'].dt, 'tz') else 'None'}")
    print(f"Primeiros valores corrigidos:")
    for i in range(min(3, len(df))):
        print(f"  {i}: {df['time'].iloc[i]}")
    
    # Salvar dataset corrigido
    print(f"\n💾 SALVANDO DATASET CORRIGIDO:")
    
    # Criar novo nome
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"GC=F_HYBRID_V2_FIXED_{timestamp}.csv"
    new_path = f"data/{new_filename}"
    
    # Salvar
    df.to_csv(new_path, index=False)
    print(f"✅ Salvo: {new_path}")
    
    # Criar backup do original
    backup_path = f"data/BACKUP_{dataset_path.split('/')[-1]}"
    import shutil
    shutil.copy2(dataset_path, backup_path)
    print(f"📦 Backup criado: {backup_path}")
    
    # Substituir original
    df.to_csv(dataset_path, index=False)
    print(f"🔄 Original substituído: {dataset_path}")
    
    print(f"\n🎉 TIMEZONE CORRIGIDO COM SUCESSO!")
    return True

if __name__ == "__main__":
    success = fix_timezone_issues()
    if success:
        print("✅ Dataset pronto para uso no silus.py")
    else:
        print("❌ Falha na correção - verificar manualmente")