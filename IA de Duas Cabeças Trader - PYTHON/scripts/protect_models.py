#!/usr/bin/env python3
"""
🔐 SCRIPT DE PROTEÇÃO DE MODELOS
===============================

Script utilitário para proteger modelos Cherry para distribuição.
Uso simples com diferentes cenários.

Exemplos de uso:
    python protect_models.py --all-cherry                    # Protege todos Cherry
    python protect_models.py --single model.zip             # Protege um modelo
    python protect_models.py --best-models                  # Protege apenas os melhores
    python protect_models.py --verify secure_model.secure   # Verifica modelo protegido
"""

import os
import sys
import argparse
from datetime import datetime

# Adicionar ao path
sys.path.append("D:/Projeto")
from trading_framework.security.secure_model_system import ModelSecurityManager, HardwareFingerprint

# 🔑 CONFIGURAÇÕES
DEFAULT_MASTER_KEY = "cherry_trading_secret_2025_v1"
CHERRY_MODELS_PATH = "D:/Projeto/Otimizacao/treino_principal/models/Cherry"

# 📋 MODELOS "MELHORES" (baseado em análises anteriores)
BEST_MODELS = [
    "Cherry_simpledirecttraining_550000_steps_20250907_232230.zip",      # 550k (baseline vencedor)
    "Cherry_simpledirecttraining_3100000_steps_20250908_031555.zip",     # 3.1M
    "Cherry_simpledirecttraining_15650000_steps_20250908_105509.zip",    # 15.65M (recente)
]


def show_system_info():
    """Mostra informações do sistema"""
    hw = HardwareFingerprint.generate()
    print(f"💻 Hardware Fingerprint: {hw}")
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def protect_all_cherry(manager, hardware_lock=True):
    """Protege todos os modelos Cherry"""
    print("🔒 PROTEÇÃO EM MASSA - TODOS OS MODELOS CHERRY")
    print("=" * 60)
    
    pattern = f"{CHERRY_MODELS_PATH}/*.zip"
    results = manager.batch_convert(pattern, hardware_lock=hardware_lock)
    
    return results


def protect_best_models(manager, hardware_lock=True):
    """Protege apenas os melhores modelos"""
    print("🏆 PROTEÇÃO SELETIVA - MELHORES MODELOS")
    print("=" * 60)
    
    results = {}
    
    for model_name in BEST_MODELS:
        model_path = os.path.join(CHERRY_MODELS_PATH, model_name)
        
        if os.path.exists(model_path):
            print(f"🔐 Protegendo: {model_name}")
            success = manager.convert_checkpoint(model_path, hardware_lock=hardware_lock)
            results[model_name] = success
            print()
        else:
            print(f"⚠️  Não encontrado: {model_name}")
            results[model_name] = False
    
    return results


def protect_single_model(manager, model_path, hardware_lock=True):
    """Protege um único modelo"""
    print(f"🔐 PROTEÇÃO INDIVIDUAL")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Arquivo não encontrado: {model_path}")
        return False
    
    success = manager.convert_checkpoint(model_path, hardware_lock=hardware_lock)
    return success


def verify_secure_model(manager, secure_path):
    """Verifica modelo protegido"""
    print(f"🔍 VERIFICAÇÃO DE MODELO PROTEGIDO")
    print("=" * 60)
    
    return manager.verify_secure_model(secure_path)


def list_available_models():
    """Lista modelos disponíveis para proteção"""
    print("📋 MODELOS DISPONÍVEIS PARA PROTEÇÃO")
    print("=" * 60)
    
    import glob
    pattern = f"{CHERRY_MODELS_PATH}/*.zip"
    models = glob.glob(pattern)
    
    if not models:
        print("❌ Nenhum modelo encontrado")
        return
    
    print(f"📍 Diretório: {CHERRY_MODELS_PATH}")
    print(f"📊 Total: {len(models)} modelos\n")
    
    # Agrupar por faixa de steps
    ranges = {
        "Early (< 1M)": [],
        "Mid (1M - 5M)": [],
        "Late (> 5M)": []
    }
    
    for model_path in sorted(models):
        filename = os.path.basename(model_path)
        
        # Extrair steps
        try:
            steps_str = filename.split('_')[2]  # Cherry_simpledirecttraining_STEPS_steps_...
            steps = int(steps_str)
            
            if steps < 1000000:
                ranges["Early (< 1M)"].append((filename, steps))
            elif steps <= 5000000:
                ranges["Mid (1M - 5M)"].append((filename, steps))
            else:
                ranges["Late (> 5M)"].append((filename, steps))
                
        except:
            ranges["Early (< 1M)"].append((filename, 0))
    
    # Mostrar por categoria
    for category, models in ranges.items():
        if models:
            print(f"📁 {category}:")
            for filename, steps in models[:5]:  # Mostrar apenas 5 por categoria
                steps_mb = steps / 1000000 if steps > 0 else 0
                print(f"   • {filename} ({steps_mb:.1f}M steps)")
            
            if len(models) > 5:
                print(f"   ... e mais {len(models) - 5} modelos")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="🔒 Sistema de Proteção de Modelos Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Proteger todos os modelos Cherry
  python protect_models.py --all-cherry

  # Proteger apenas os melhores modelos
  python protect_models.py --best-models

  # Proteger um modelo específico
  python protect_models.py --single "path/to/model.zip"

  # Proteger sem hardware lock (para distribuição)
  python protect_models.py --all-cherry --no-hardware-lock

  # Verificar modelo protegido
  python protect_models.py --verify "model.secure"

  # Listar modelos disponíveis
  python protect_models.py --list
        """
    )
    
    # Opções de proteção
    parser.add_argument('--all-cherry', action='store_true',
                       help='Protege todos os modelos Cherry')
    
    parser.add_argument('--best-models', action='store_true',
                       help='Protege apenas os melhores modelos (curated list)')
    
    parser.add_argument('--single', metavar='MODEL_PATH',
                       help='Protege um único modelo')
    
    parser.add_argument('--verify', metavar='SECURE_PATH',
                       help='Verifica modelo protegido')
    
    parser.add_argument('--list', action='store_true',
                       help='Lista modelos disponíveis')
    
    # Opções de configuração
    parser.add_argument('--master-key', default=DEFAULT_MASTER_KEY,
                       help=f'Chave mestra (padrão: {DEFAULT_MASTER_KEY})')
    
    parser.add_argument('--no-hardware-lock', action='store_true',
                       help='Desativa hardware lock (para distribuição)')
    
    args = parser.parse_args()
    
    # Header
    print("🔒 SISTEMA DE PROTEÇÃO DE MODELOS TRADING")
    print("=" * 60)
    show_system_info()
    
    # Lista modelos se solicitado
    if args.list:
        list_available_models()
        return
    
    # Hardware lock setting
    hardware_lock = not args.no_hardware_lock
    
    # Criar manager
    manager = ModelSecurityManager(args.master_key)
    
    # Executar ação solicitada
    if args.all_cherry:
        results = protect_all_cherry(manager, hardware_lock)
        
    elif args.best_models:
        results = protect_best_models(manager, hardware_lock)
        
    elif args.single:
        success = protect_single_model(manager, args.single, hardware_lock)
        results = {os.path.basename(args.single): success}
        
    elif args.verify:
        success = verify_secure_model(manager, args.verify)
        return
        
    else:
        print("❌ Nenhuma ação especificada. Use --help para ver opções.")
        return
    
    # Mostrar resumo final
    if 'results' in locals():
        successful = sum(results.values())
        total = len(results)
        
        print("\n" + "=" * 60)
        print("📊 RESUMO FINAL")
        print(f"✅ Sucessos: {successful}/{total}")
        print(f"🔑 Hardware Lock: {'Ativo' if hardware_lock else 'Desativo'}")
        print(f"🔐 Master Key: {args.master_key[:10]}...")
        
        if successful > 0:
            print(f"\n🎉 {successful} modelo(s) protegido(s) com sucesso!")
            print("💡 Use --verify para testar modelos protegidos")


if __name__ == "__main__":
    main()