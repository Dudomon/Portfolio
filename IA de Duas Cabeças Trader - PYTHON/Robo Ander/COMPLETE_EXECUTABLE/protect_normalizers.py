#!/usr/bin/env python3
"""
🔒 PROTEÇÃO DE NORMALIZERS
=========================

Script para proteger arquivos .pkl de normalizers usando criptografia
"""

import sys
import os
import pickle
from cryptography.fernet import Fernet
import hashlib

# Adicionar ao path
sys.path.append("D:/Projeto")
from trading_framework.security.secure_model_system import HardwareFingerprint

def generate_key_from_master(master_key: str) -> bytes:
    """Gera chave Fernet a partir da master key"""
    key_hash = hashlib.sha256(master_key.encode()).digest()
    return Fernet.generate_key()

def protect_normalizer(input_path: str, master_key: str, hardware_lock: bool = True):
    """Protege um normalizer .pkl"""
    if not os.path.exists(input_path):
        print(f"❌ Arquivo não encontrado: {input_path}")
        return False
        
    output_path = input_path.replace('.pkl', '.secure_pkl')
    
    try:
        print(f"🔐 Protegendo normalizer: {os.path.basename(input_path)}")
        
        # 1. Carregar normalizer original
        with open(input_path, 'rb') as f:
            normalizer_data = f.read()
        
        # 2. Preparar dados para proteção
        protected_data = {
            'normalizer_data': normalizer_data,
            'filename': os.path.basename(input_path),
            'hardware_fingerprint': HardwareFingerprint.generate() if hardware_lock else None,
            'protected_at': __import__('time').time(),
            'master_key_hash': hashlib.sha256(master_key.encode()).hexdigest()
        }
        
        # 3. Serializar
        serialized = pickle.dumps(protected_data, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 4. Criptografar
        key_material = hashlib.sha256(master_key.encode()).digest()[:32]
        fernet_key = Fernet.generate_key()
        cipher = Fernet(fernet_key)
        encrypted_data = cipher.encrypt(serialized)
        
        # 5. Salvar com chave
        final_data = {
            'encrypted_normalizer': encrypted_data,
            'key': fernet_key,
            'key_derived_from': hashlib.sha256(master_key.encode()).hexdigest()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"   ✅ Normalizer protegido: {output_path}")
        print(f"   📦 Tamanho: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro ao proteger normalizer: {e}")
        return False

def load_protected_normalizer(input_path: str, master_key: str):
    """Carrega um normalizer protegido"""
    try:
        print(f"🔓 Carregando normalizer protegido: {os.path.basename(input_path)}")
        
        # 1. Carregar dados criptografados
        with open(input_path, 'rb') as f:
            encrypted_package = pickle.load(f)
        
        # 2. Verificar chave master
        expected_hash = hashlib.sha256(master_key.encode()).hexdigest()
        if encrypted_package['key_derived_from'] != expected_hash:
            raise ValueError("❌ Master key incorreta")
        
        # 3. Descriptografar
        cipher = Fernet(encrypted_package['key'])
        decrypted_data = cipher.decrypt(encrypted_package['encrypted_normalizer'])
        protected_data = pickle.loads(decrypted_data)
        
        # 4. Verificar hardware lock (se ativo)
        if protected_data['hardware_fingerprint']:
            current_fingerprint = HardwareFingerprint.generate()
            if current_fingerprint != protected_data['hardware_fingerprint']:
                raise ValueError("❌ Hardware não autorizado")
            print("   ✅ Hardware autorizado")
        
        # 5. Extrair normalizer
        normalizer_data = protected_data['normalizer_data']
        normalizer = pickle.loads(normalizer_data)
        
        print(f"   ✅ Normalizer carregado: {protected_data['filename']}")
        return normalizer
        
    except Exception as e:
        print(f"   ❌ Erro ao carregar normalizer: {e}")
        raise

def main():
    """Protege os normalizers do Legion V1"""
    print("🔒 PROTEÇÃO DE NORMALIZERS - LEGION V1")
    print("=" * 50)
    
    master_key = "cherry_trading_secret_2025_v1"
    base_path = "D:/Projeto/Modelo PPO Trader/Modelo daytrade"
    
    normalizers = [
        "enhanced_normalizer_final.pkl",
        "enhanced_normalizer_final_enhanced.pkl"
    ]
    
    results = []
    
    for normalizer in normalizers:
        full_path = os.path.join(base_path, normalizer)
        success = protect_normalizer(full_path, master_key, hardware_lock=True)
        results.append((normalizer, success))
    
    print("\n" + "="*50)
    print("📊 RESUMO DA PROTEÇÃO")
    print("="*50)
    
    for normalizer, success in results:
        status = "✅ PROTEGIDO" if success else "❌ FALHOU"
        print(f"{status} - {normalizer}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 {successful}/{total} normalizers protegidos com sucesso")
    
    if successful > 0:
        print("\n💡 Normalizers protegidos podem ser carregados com:")
        print("   normalizer = load_protected_normalizer('file.secure_pkl', master_key)")

if __name__ == "__main__":
    main()