#!/usr/bin/env python3
"""
🔍 DEBUG SL POSITIONS - Verificar se posições têm SL definido
"""

import re

def extract_position_logs():
    """Extrair logs de posições do arquivo de avaliação"""
    
    with open("avaliacoes/avaliacao_v7_2000k_20250820_110353.txt", "r") as f:
        content = f.read()
    
    print("🔍 ANALISANDO LOGS DE AVALIAÇÃO...")
    
    # Encontrar trades com perdas > 300
    large_loss_pattern = r"Trade #\d+.*PnL=\$-([3-9]\d{2}|\d{4})"
    large_losses = re.findall(large_loss_pattern, content)
    
    if large_losses:
        print(f"🚨 ENCONTRADAS {len(large_losses)} PERDAS > $300:")
        for loss in large_losses[:5]:  # Primeiras 5
            print(f"   Perda: ${loss}")
    
    # Procurar por logs de posições com SL
    position_pattern = r"NEW POSITION OPENED.*"
    positions = re.findall(position_pattern, content)
    
    if positions:
        print(f"\n📊 POSIÇÕES ABERTAS: {len(positions)}")
        for pos in positions[:3]:  # Primeiras 3
            print(f"   {pos}")
    else:
        print("❌ Nenhuma posição encontrada nos logs")
    
    # Procurar por SL hits
    sl_pattern = r"SL hit|stop.*loss"
    sl_hits = re.findall(sl_pattern, content, re.IGNORECASE)
    
    print(f"\n🎯 SL HITS ENCONTRADOS: {len(sl_hits)}")
    
    # Calcular taxa de SL
    total_trades_pattern = r"Trade #(\d+)"
    all_trades = re.findall(total_trades_pattern, content)
    total_trades = len(set(all_trades))
    
    if total_trades > 0:
        sl_rate = (len(sl_hits) / total_trades) * 100
        print(f"📈 TAXA DE SL: {sl_rate:.1f}% ({len(sl_hits)}/{total_trades})")
        
        if sl_rate < 10:  # Menos de 10% dos trades teve SL
            print("🚨 TAXA DE SL MUITO BAIXA - POSSÍVEL BUG!")
    
    return large_losses, sl_hits, total_trades

if __name__ == "__main__":
    try:
        large_losses, sl_hits, total_trades = extract_position_logs()
        
        if len(large_losses) > 0:
            print("\n🔴 BUG CONFIRMADO: Perdas impossíveis detectadas!")
            print("   - SL máximo deveria ser ~$40")  
            print(f"   - Perdas reais: ${large_losses[0]}+")
            print("   - SISTEMA DE SL NÃO ESTÁ FUNCIONANDO!")
        else:
            print("\n🟢 Nenhuma perda excessiva encontrada")
            
    except FileNotFoundError:
        print("❌ Arquivo de avaliação não encontrado")
    except Exception as e:
        print(f"❌ Erro na análise: {e}")