#!/usr/bin/env python3
"""
🧪 TESTE DE DETECÇÃO DE SÍMBOLOS
================================

Testa a funcionalidade de detecção automática de símbolos do ouro
"""

import MetaTrader5 as mt5
import sys
import os

# Adicionar o diretório atual ao path para importar RobotV7
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_symbol_detection():
    """Testa apenas a detecção de símbolos sem inicializar toda a classe"""
    
    print("🧪 TESTE DE DETECÇÃO DE SÍMBOLOS DO OURO")
    print("=" * 50)
    
    # Lista de símbolos para testar
    gold_symbols = [
        "XAUUSDz",    # Exness Zero accounts
        "XAUUSD",     # Standard brokers
        "xauusd",     # Lowercase variant
        "GOLD",       # Some brokers use GOLD
        "gold#",      # Hash variant
        "Gold",       # Capitalized
        "XAU/USD",    # Slash notation
        "XAUUSD.a",   # Admiral Markets
        "XAUUSD-Z",   # Zero suffix variant
        "_XAUUSD"     # Underscore prefix
    ]
    
    # Tentar inicializar MT5
    print("\n[🔌 CONNECTING] Inicializando MT5...")
    
    if not mt5.initialize():
        print(f"❌ Falha ao inicializar MT5. Erro: {mt5.last_error()}")
        return None
    
    print("✅ MT5 inicializado com sucesso")
    
    # Testar cada símbolo
    found_symbols = []
    
    print(f"\n[🔍 DETECTION] Testando {len(gold_symbols)} variações de símbolos...")
    
    for i, symbol in enumerate(gold_symbols, 1):
        print(f"\n[{i:2d}/{len(gold_symbols)}] Testando: {symbol}")
        
        try:
            # Tentar selecionar o símbolo
            if mt5.symbol_select(symbol, True):
                # Verificar se o símbolo tem informações válidas
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info and symbol_info.visible:
                    # Verificar se há dados de preço
                    tick = mt5.symbol_info_tick(symbol)
                    if tick and tick.bid > 0 and tick.ask > 0:
                        print(f"  ✅ VÁLIDO - Bid: {tick.bid} | Ask: {tick.ask}")
                        print(f"    Spread: {symbol_info.spread} | Digits: {symbol_info.digits}")
                        print(f"    Min lot: {symbol_info.volume_min} | Max lot: {symbol_info.volume_max}")
                        found_symbols.append({
                            'symbol': symbol,
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'spread': symbol_info.spread,
                            'digits': symbol_info.digits,
                            'volume_min': symbol_info.volume_min,
                            'volume_max': symbol_info.volume_max
                        })
                    else:
                        print(f"  ❌ SEM DADOS - Sem preços válidos")
                else:
                    print(f"  ❌ NÃO VISÍVEL - Não está no Market Watch")
            else:
                print(f"  ❌ NÃO ENCONTRADO - Símbolo inexistente")
                
        except Exception as e:
            print(f"  ❌ ERRO - {e}")
    
    # Mostrar resultados
    print(f"\n{'='*50}")
    print(f"📊 RESUMO DOS RESULTADOS")
    print(f"{'='*50}")
    
    if found_symbols:
        print(f"✅ Encontrados {len(found_symbols)} símbolo(s) válido(s):")
        
        for i, sym in enumerate(found_symbols, 1):
            print(f"\n[{i}] {sym['symbol']}")
            print(f"    Preços: {sym['bid']:.5f} / {sym['ask']:.5f}")
            print(f"    Spread: {sym['spread']} | Digits: {sym['digits']}")
            print(f"    Volume: {sym['volume_min']:.2f} - {sym['volume_max']:.2f}")
        
        # Recomendar o primeiro encontrado
        recommended = found_symbols[0]['symbol']
        print(f"\n🏅 RECOMENDADO: {recommended}")
        
    else:
        print(f"❌ Nenhum símbolo válido encontrado!")
        recommended = None
    
    # Cleanup
    mt5.shutdown()
    print(f"\n🔌 MT5 desconectado")
    
    return recommended

def test_robotv7_integration():
    """Testa a integração com RobotV7 (se possível)"""
    print(f"\n🤖 TESTE DE INTEGRAÇÃO COM ROBOTV7")
    print("=" * 50)
    
    try:
        # Tentar importar TradingRobotV7
        from RobotV7 import TradingRobotV7, Config
        
        print(f"✅ RobotV7 importado com sucesso")
        print(f"📋 Símbolos configurados: {Config.GOLD_SYMBOLS}")
        
        # Tentar criar instância (apenas para teste de importação)
        print(f"\n⚠️  NOTA: Para teste completo, execute o RobotV7 diretamente")
        
    except ImportError as e:
        print(f"❌ Erro ao importar RobotV7: {e}")
    except Exception as e:
        print(f"❌ Erro geral: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO TESTE DE DETECÇÃO DE SÍMBOLOS")
    
    # Teste standalone da detecção
    result = test_symbol_detection()
    
    # Teste de integração
    test_robotv7_integration()
    
    print(f"\n🏁 TESTE FINALIZADO")
    if result:
        print(f"🎯 Símbolo recomendado: {result}")
    else:
        print(f"⚠️  Nenhum símbolo encontrado - verifique conexão MT5")