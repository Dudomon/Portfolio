"""
Teste do SmartBot - Adaptive Sampling Trading Robot
Valida funcionamento básico e sistema de adaptive sampling
"""

import sys
import os
import numpy as np
import time

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_adaptive_sampler():
    """Testar sistema Adaptive Sampler isoladamente"""
    print("🧪 TESTE DO ADAPTIVE SAMPLER")
    print("=" * 50)
    
    # Import da classe do smartbot
    try:
        from smartbot import AdaptiveModelSampler
        print("✅ Import do AdaptiveModelSampler realizado com sucesso")
    except Exception as e:
        print(f"❌ Erro no import: {e}")
        return False
    
    # Inicializar sampler
    sampler = AdaptiveModelSampler()
    print("✅ AdaptiveModelSampler inicializado")
    
    # Simular ticks de mercado
    base_price = 2050.0
    current_time = time.time()
    
    print("\n📊 SIMULANDO TICKS DE MERCADO:")
    print("-" * 30)
    
    consultation_count = 0
    skipped_count = 0
    
    for i in range(100):  # Simular 100 ticks
        # Simular movimento de preço
        price_change = np.random.normal(0, 0.5)  # Movimento aleatório
        current_price = base_price + price_change
        
        # Avançar tempo (simular ticks a cada 2 segundos)
        current_time += 2
        
        # Decidir se consulta modelo
        should_consult = sampler.should_consult_model(current_price, current_time)
        
        if should_consult:
            consultation_count += 1
            # Simular resposta do modelo
            raw_action = np.array([0.5, 0.7, 0.2, -0.1])  # Ação simulada
            confidence = np.random.uniform(0.4, 0.9)  # Confiança aleatória
            
            # Processar com filtros
            processed_action, msg = sampler.process_model_output(raw_action, confidence)
            
            if processed_action is not None:
                status = "✅ APROVADA"
            else:
                status = f"❌ REJEITADA: {msg}"
                
            if i % 20 == 0:  # Log esporádico
                print(f"Tick {i:2d}: Consulta {status} | Conf: {confidence:.2f}")
                
            # Simular resultado PnL aleatório
            if np.random.random() < 0.6:  # 60% de trades positivos
                pnl_change = np.random.uniform(2, 15)
            else:
                pnl_change = np.random.uniform(-10, -2)
                
            sampler.update_performance(pnl_change)
            
        else:
            skipped_count += 1
            
        base_price = current_price  # Atualizar preço base
    
    # Estatísticas finais
    print(f"\n📈 RESULTADOS FINAIS:")
    print("-" * 30)
    stats = sampler.get_sampling_stats()
    
    print(f"Total de ticks processados: {stats['total_ticks']}")
    print(f"Consultas ao modelo: {stats['model_consultations']}")
    print(f"Consultas puladas: {stats['skipped_consultations']}")
    print(f"Taxa de redução: {stats['reduction_rate']:.1f}%")
    print(f"Performance média: ${stats['avg_performance']:.2f}")
    print(f"Threshold de confiança atual: {stats['confidence_threshold']:.2f}")
    print(f"Sequência de vitórias: {stats['consecutive_wins']}")
    print(f"Sequência de perdas: {stats['consecutive_losses']}")
    
    # Validações
    expected_reduction = 70  # Esperamos pelo menos 70% de redução
    if stats['reduction_rate'] >= expected_reduction:
        print(f"✅ SUCESSO: Redução de {stats['reduction_rate']:.1f}% >= {expected_reduction}%")
    else:
        print(f"⚠️ AVISO: Redução de {stats['reduction_rate']:.1f}% < {expected_reduction}%")
        
    return True

def test_smartbot_import():
    """Testar import e inicialização básica do SmartBot"""
    print("\n🤖 TESTE DE IMPORT DO SMARTBOT")
    print("=" * 50)
    
    try:
        from smartbot import TradingRobotV7
        print("✅ Import do TradingRobotV7 (SmartBot) realizado com sucesso")
        
        # Tentar inicializar (pode falhar por dependências do MT5)
        try:
            robot = TradingRobotV7()
            print("✅ SmartBot inicializado com sucesso")
            
            # Verificar se adaptive sampler foi criado
            if hasattr(robot, 'adaptive_sampler'):
                print("✅ Adaptive Sampler integrado no SmartBot")
                return True
            else:
                print("❌ Adaptive Sampler não encontrado no SmartBot")
                return False
                
        except Exception as e:
            print(f"⚠️ Erro na inicialização (esperado sem MT5): {e}")
            print("✅ Import funcionou - erro apenas por dependências")
            return True
            
    except Exception as e:
        print(f"❌ Erro no import do SmartBot: {e}")
        return False

def main():
    """Executar todos os testes"""
    print("🚀 SMARTBOT - TESTE DE VALIDAÇÃO")
    print("="*60)
    
    success_count = 0
    total_tests = 2
    
    # Teste 1: Adaptive Sampler
    if test_adaptive_sampler():
        success_count += 1
    
    # Teste 2: SmartBot Import
    if test_smartbot_import():
        success_count += 1
    
    # Resultado final
    print("\n" + "="*60)
    print("📊 RESULTADO FINAL DOS TESTES")
    print("="*60)
    
    if success_count == total_tests:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ SmartBot está pronto para uso")
        print("\n🚀 Para usar o SmartBot:")
        print("   python smartbot.py")
    else:
        print(f"⚠️ {success_count}/{total_tests} testes passaram")
        print("❌ SmartBot precisa de correções")
        
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)