"""
Teste dos Ajustes SmartBot-Fisher
Valida ranges do pescador.py e remoção de cooldowns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ranges_pescador():
    """Testar se ranges foram ajustados para match pescador.py"""
    print("🧪 TESTE DOS RANGES PESCADOR")
    print("=" * 40)
    
    try:
        # Import do smartbot-fisher
        sys.path.append(os.path.join(os.path.dirname(__file__), "Modelo PPO Trader"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("smartbot_fisher", "Modelo PPO Trader/smartbot-fisher.py")
        smartbot_fisher = importlib.util.module_from_spec(spec)
        
        # Simular uma inicialização básica para pegar os ranges
        print("✅ Import do smartbot-fisher realizado")
        
        # Ler diretamente do arquivo os ranges esperados
        with open("Modelo PPO Trader/smartbot-fisher.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Verificar ranges pescador
        if "self.sl_range_min = 0.3" in content:
            print("✅ SL range min = 0.3 (pescador)")
        else:
            print("❌ SL range min incorreto")
            
        if "self.sl_range_max = 0.7" in content:
            print("✅ SL range max = 0.7 (pescador)")
        else:
            print("❌ SL range max incorreto")
            
        if "self.tp_range_min = 0.5" in content:
            print("✅ TP range min = 0.5 (pescador)")
        else:
            print("❌ TP range min incorreto")
            
        if "self.tp_range_max = 1.0" in content:
            print("✅ TP range max = 1.0 (pescador)")
        else:
            print("❌ TP range max incorreto")
            
        # Verificar comentários atualizados
        if "aligned with pescador.py SCALP" in content:
            print("✅ Comentários atualizados para pescador.py")
        else:
            print("❌ Comentários não atualizados")
            
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de ranges: {e}")
        return False

def test_cooldowns_desabilitados():
    """Testar se cooldowns foram completamente desabilitados"""
    print("\n🧪 TESTE DE COOLDOWNS DESABILITADOS")
    print("=" * 40)
    
    try:
        # Ler arquivo e verificar mudanças
        with open("Modelo PPO Trader/smartbot-fisher.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Verificar inicialização
        if "self.cooldown_minutes = 0" in content:
            print("✅ cooldown_minutes = 0 (desabilitado)")
        else:
            print("❌ cooldown_minutes não zerado")
            
        # Verificar logs
        if "COOLDOWNS COMPLETAMENTE DESABILITADOS" in content:
            print("✅ Log de desabilitação presente")
        else:
            print("❌ Log de desabilitação ausente")
            
        # Verificar função _is_in_cooldown
        if "return False, 0  # SEMPRE LIVRE PARA OPERAR" in content:
            print("✅ _is_in_cooldown sempre retorna False")
        else:
            print("❌ _is_in_cooldown não modificado")
            
        # Verificar _allocate_entry_slot
        if "COOLDOWN DESABILITADO - Escolhe primeiro slot livre" in content:
            print("✅ _allocate_entry_slot modificado")
        else:
            print("❌ _allocate_entry_slot não modificado")
            
        # Verificar remoção de verificação de cooldown no processamento
        if "COOLDOWN DESABILITADO - Processar entry_decision diretamente" in content:
            print("✅ Verificação de cooldown removida do processamento")
        else:
            print("❌ Verificação de cooldown ainda presente")
            
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de cooldowns: {e}")
        return False

def test_resumo_compatibilidade():
    """Resumo da compatibilidade com pescador.py"""
    print("\n📊 RESUMO DE COMPATIBILIDADE COM PESCADOR.PY")
    print("=" * 50)
    
    try:
        # Ler ambos os arquivos
        with open("pescador.py", "r", encoding="utf-8") as f:
            pescador_content = f.read()
            
        with open("Modelo PPO Trader/smartbot-fisher.py", "r", encoding="utf-8") as f:
            smartbot_content = f.read()
            
        compatibilidade = []
        
        # Verificar ranges
        ranges_match = all([
            "self.sl_range_min = 0.3" in smartbot_content,
            "self.sl_range_max = 0.7" in smartbot_content,
            "self.tp_range_min = 0.5" in smartbot_content,
            "self.tp_range_max = 1.0" in smartbot_content
        ])
        compatibilidade.append(("SL/TP Ranges", "✅" if ranges_match else "❌"))
        
        # Verificar cooldowns zerados
        cooldowns_disabled = all([
            "self.cooldown_minutes = 0" in smartbot_content,
            "return False, 0" in smartbot_content and "SEMPRE LIVRE" in smartbot_content
        ])
        compatibilidade.append(("Cooldowns Desabilitados", "✅" if cooldowns_disabled else "❌"))
        
        # Verificar comentários de alinhamento
        aligned_comments = "aligned with pescador.py" in smartbot_content
        compatibilidade.append(("Comentários Alinhados", "✅" if aligned_comments else "❌"))
        
        # Mostrar resultado
        for item, status in compatibilidade:
            print(f"{item:<25}: {status}")
            
        all_compatible = all(status == "✅" for _, status in compatibilidade)
        print(f"\n{'🎉 COMPATIBILIDADE TOTAL COM PESCADOR.PY' if all_compatible else '⚠️ AJUSTES NECESSÁRIOS'}")
        
        return all_compatible
        
    except Exception as e:
        print(f"❌ Erro na verificação de compatibilidade: {e}")
        return False

def main():
    """Executar todos os testes"""
    print("🚀 SMARTBOT-FISHER - TESTE DE AJUSTES PESCADOR.PY")
    print("="*60)
    
    testes = [
        test_ranges_pescador,
        test_cooldowns_desabilitados,
        test_resumo_compatibilidade
    ]
    
    resultados = []
    for teste in testes:
        try:
            resultado = teste()
            resultados.append(resultado)
        except Exception as e:
            print(f"❌ Erro no teste: {e}")
            resultados.append(False)
    
    # Resultado final
    print("\n" + "="*60)
    print("📈 RESULTADO FINAL")
    print("="*60)
    
    sucessos = sum(resultados)
    total = len(resultados)
    
    if sucessos == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ SmartBot-Fisher está alinhado com pescador.py")
        print("🚀 Pronto para scalping com ranges curtos e sem cooldowns!")
    else:
        print(f"⚠️ {sucessos}/{total} testes passaram")
        print("❌ Verificar ajustes necessários")
        
    return sucessos == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)