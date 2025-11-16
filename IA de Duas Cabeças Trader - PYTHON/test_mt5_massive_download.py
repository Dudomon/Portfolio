#!/usr/bin/env python3
"""
🔍 TESTE DE CAPACIDADE MT5 - 1 MILHÃO DE BARRAS 1MIN
===================================================

Verificar se MT5 consegue fornecer grandes volumes de dados históricos
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_mt5_massive_capacity():
    """🔍 TESTE DE CAPACIDADE MASSIVA DO MT5"""

    logger = setup_logging()

    logger.info("🔍 TESTE DE CAPACIDADE MASSIVA MT5 - 1 MILHÃO DE BARRAS")
    logger.info("=" * 80)

    # Conectar MT5
    if not mt5.initialize():
        logger.error("❌ Falha ao conectar MT5")
        logger.error(f"Erro: {mt5.last_error()}")
        return

    logger.info("✅ MT5 conectado com sucesso")

    # Símbolos de ouro para testar
    gold_symbols = ["GOLD", "XAUUSD", "GOLD#", "GOLDUSD", "XAU/USD"]

    # Testar diferentes períodos históricos
    test_periods = [
        ("3 meses", timedelta(days=90)),
        ("6 meses", timedelta(days=180)),
        ("1 ano", timedelta(days=365)),
        ("2 anos", timedelta(days=730)),
        ("3 anos", timedelta(days=1095)),
    ]

    results = {}
    best_symbol = None
    max_bars = 0

    for symbol in gold_symbols:
        logger.info(f"\n🔍 TESTANDO SÍMBOLO: {symbol}")
        logger.info("-" * 50)

        try:
            # Verificar se símbolo existe
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"❌ {symbol}: Símbolo não encontrado")
                continue

            if not symbol_info.visible:
                logger.info(f"🔧 {symbol}: Ativando símbolo")
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"❌ {symbol}: Falha ao ativar")
                    continue

            symbol_results = {}

            # Testar diferentes períodos
            for period_name, delta in test_periods:
                try:
                    end_date = datetime.now()
                    start_date = end_date - delta

                    logger.info(f"  📅 Testando {period_name}: {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")

                    # Tentar baixar dados
                    rates = mt5.copy_rates_range(
                        symbol,
                        mt5.TIMEFRAME_M1,
                        start_date,
                        end_date
                    )

                    if rates is None or len(rates) == 0:
                        logger.warning(f"    ❌ {period_name}: Nenhum dado obtido")
                        symbol_results[period_name] = 0
                        continue

                    bars_count = len(rates)
                    symbol_results[period_name] = bars_count

                    # Calcular estatísticas básicas
                    df_temp = pd.DataFrame(rates)
                    df_temp['time'] = pd.to_datetime(df_temp['time'], unit='s')

                    actual_days = (df_temp['time'].max() - df_temp['time'].min()).days
                    bars_per_day = bars_count / max(actual_days, 1)

                    logger.info(f"    ✅ {period_name}: {bars_count:,} barras ({actual_days} dias, {bars_per_day:.0f} barras/dia)")

                    # Atualizar máximo
                    if bars_count > max_bars:
                        max_bars = bars_count
                        best_symbol = symbol

                except Exception as e:
                    logger.error(f"    ❌ {period_name}: Erro - {e}")
                    symbol_results[period_name] = 0

            results[symbol] = symbol_results

        except Exception as e:
            logger.error(f"❌ {symbol}: Erro geral - {e}")
            continue

    # Finalizar MT5
    mt5.shutdown()

    # RELATÓRIO FINAL
    logger.info(f"\n" + "=" * 80)
    logger.info("📊 RELATÓRIO FINAL - CAPACIDADE MT5")
    logger.info("=" * 80)

    logger.info(f"\n🏆 MELHOR PERFORMANCE:")
    logger.info(f"   Símbolo: {best_symbol}")
    logger.info(f"   Máximo de barras: {max_bars:,}")

    # Estimar quantos anos seriam necessários para 1 milhão
    if max_bars > 0:
        ratio_needed = 1_000_000 / max_bars
        best_period_days = max([delta.days for _, delta in test_periods])
        years_needed = (best_period_days * ratio_needed) / 365

        logger.info(f"   Para 1M barras seriam necessários: {years_needed:.1f} anos")

        if max_bars >= 1_000_000:
            logger.info(f"   🎯 PARABÉNS! MT5 consegue fornecer 1M+ barras!")
        elif max_bars >= 500_000:
            logger.info(f"   ✅ MUITO BOM! MT5 consegue fornecer 500K+ barras")
        elif max_bars >= 250_000:
            logger.info(f"   👍 BOM! MT5 consegue fornecer 250K+ barras")
        else:
            logger.info(f"   ⚠️ LIMITADO: MT5 tem restrições de histórico")

    # Tabela detalhada
    logger.info(f"\n📋 TABELA DETALHADA:")
    logger.info(f"{'Símbolo':<10} {'3 meses':<10} {'6 meses':<10} {'1 ano':<10} {'2 anos':<10} {'3 anos':<10}")
    logger.info("-" * 70)

    for symbol, symbol_results in results.items():
        row = f"{symbol:<10}"
        for period_name, _ in test_periods:
            count = symbol_results.get(period_name, 0)
            if count > 0:
                row += f" {count/1000:.0f}K".rjust(10)
            else:
                row += f" {'N/A':<10}"
        logger.info(row)

    # RECOMENDAÇÕES
    logger.info(f"\n🎯 RECOMENDAÇÕES:")

    if max_bars >= 800_000:
        logger.info("   ✅ MT5 pode fornecer dataset massivo suficiente!")
        logger.info("   📝 Usar 2-3 anos de dados para treino robusto")
    elif max_bars >= 400_000:
        logger.info("   👍 MT5 pode fornecer dataset grande")
        logger.info("   📝 Usar 1-2 anos de dados, complementar se necessário")
    else:
        logger.info("   ⚠️ MT5 tem limitações de histórico")
        logger.info("   📝 Considerar combinar com outras fontes")

    logger.info(f"\n📊 MELHOR ESTRATÉGIA:")
    if best_symbol and max_bars > 0:
        logger.info(f"   1. Usar símbolo: {best_symbol}")
        logger.info(f"   2. Baixar máximo disponível: {max_bars:,} barras")
        logger.info(f"   3. Período recomendado: 2-3 anos")
        logger.info(f"   4. Qualidade esperada: Dados reais MT5")

    logger.info(f"\n✅ TESTE DE CAPACIDADE CONCLUÍDO!")

    return results, best_symbol, max_bars

if __name__ == "__main__":
    test_mt5_massive_capacity()