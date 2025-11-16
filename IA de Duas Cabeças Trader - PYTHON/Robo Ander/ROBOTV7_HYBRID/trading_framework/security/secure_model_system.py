"""
🔒 SISTEMA DE PROTEÇÃO DE MODELOS TRADING
==========================================

Sistema completo para proteger modelos PPO contra:
- Uso não autorizado
- Extração de pesos
- Execução em hardware não autorizado
- Reverse engineering

Autor: Claude Code
Data: 2025-09-08
"""

import os
import pickle
import hashlib
import time
import base64
import json
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
import torch
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO


class HardwareFingerprint:
    """Gera fingerprint único do hardware para lock"""
    
    @staticmethod
    def get_cpu_info():
        """Obtém informações do CPU"""
        try:
            import cpuinfo
            return cpuinfo.get_cpu_info()['brand_raw']
        except:
            # Fallback para Windows
            import platform
            return f"{platform.processor()}_{platform.machine()}"
    
    @staticmethod
    def get_gpu_info():
        """Obtém informações da GPU"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
            else:
                return "no_gpu"
        except:
            return "unknown_gpu"
    
    @classmethod
    def generate(cls) -> str:
        """Gera fingerprint único da máquina"""
        cpu = cls.get_cpu_info()
        gpu = cls.get_gpu_info()
        
        # Criar hash único mas estável
        hardware_string = f"{cpu}_{gpu}"
        fingerprint = hashlib.sha256(hardware_string.encode()).hexdigest()[:16]
        
        return fingerprint


class ModelObfuscator:
    """Sistema de obfuscação reversível dos pesos do modelo"""
    
    @staticmethod
    def _generate_key_sequence(secret_key: str, length: int) -> np.ndarray:
        """Gera sequência pseudo-aleatória baseada na chave"""
        seed = int(hashlib.sha256(secret_key.encode()).hexdigest()[:8], 16)
        np.random.seed(seed % (2**31))
        return np.random.randint(0, 65536, size=length)
    
    @staticmethod
    def obfuscate_tensor(tensor: torch.Tensor, secret_key: str) -> torch.Tensor:
        """Obfusca um tensor usando XOR + rotação"""
        if tensor.numel() == 0:
            return tensor
            
        # Gerar chaves
        keys = ModelObfuscator._generate_key_sequence(secret_key, tensor.numel())
        
        # Flatten tensor
        flat_tensor = tensor.flatten()
        
        # XOR com chaves
        for i in range(len(flat_tensor)):
            if flat_tensor[i].dtype in [torch.float32, torch.float64]:
                # Para floats, aplicar transformação matemática
                flat_tensor[i] = flat_tensor[i] * (1 + (keys[i] % 1000) / 10000.0)
        
        # Rotação circular
        shift = keys[0] % tensor.numel()
        flat_tensor = torch.roll(flat_tensor, shift, dims=0)
        
        return flat_tensor.reshape(tensor.shape)
    
    @staticmethod
    def deobfuscate_tensor(tensor: torch.Tensor, secret_key: str) -> torch.Tensor:
        """Reverte obfuscação do tensor"""
        if tensor.numel() == 0:
            return tensor
            
        # Gerar mesmas chaves
        keys = ModelObfuscator._generate_key_sequence(secret_key, tensor.numel())
        
        # Flatten tensor
        flat_tensor = tensor.flatten()
        
        # Reverter rotação
        shift = keys[0] % tensor.numel()
        flat_tensor = torch.roll(flat_tensor, -shift, dims=0)
        
        # Reverter XOR
        for i in range(len(flat_tensor)):
            if flat_tensor[i].dtype in [torch.float32, torch.float64]:
                flat_tensor[i] = flat_tensor[i] / (1 + (keys[i] % 1000) / 10000.0)
        
        return flat_tensor.reshape(tensor.shape)


class SecureModelWrapper:
    """Wrapper principal para modelos seguros"""
    
    def __init__(self, model_type: str = "PPO"):
        self.model_type = model_type
        self.version = "1.0"
        self.created_at = time.time()
        
    def save_secure(self, 
                   model: Any, 
                   output_path: str, 
                   master_key: str,
                   hardware_lock: bool = True,
                   additional_info: Optional[Dict] = None) -> bool:
        """
        Salva modelo em formato seguro
        
        Args:
            model: Modelo PPO/RecurrentPPO treinado
            output_path: Caminho para salvar arquivo .secure
            master_key: Chave mestra para criptografia
            hardware_lock: Se deve travar no hardware atual
            additional_info: Informações adicionais (metadata, etc.)
        """
        try:
            print(f"🔐 Protegendo modelo: {os.path.basename(output_path)}")
            
            # 1. Extrair state dict
            if hasattr(model, 'get_parameters'):
                state_dict = model.get_parameters()
            else:
                state_dict = model.policy.state_dict()
            
            # 2. Obfuscar pesos
            print("   🔄 Obfuscando pesos...")
            obfuscated_state = {}
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    obfuscated_state[key] = ModelObfuscator.obfuscate_tensor(tensor, master_key)
                else:
                    obfuscated_state[key] = tensor
            
            # 3. Preparar dados para criptografia
            model_data = {
                'state_dict': obfuscated_state,
                'model_type': self.model_type,
                'version': self.version,
                'created_at': self.created_at,
                'hardware_fingerprint': HardwareFingerprint.generate() if hardware_lock else None,
                'additional_info': additional_info or {}
            }
            
            # 4. Serializar
            print("   📦 Serializando dados...")
            serialized_data = pickle.dumps(model_data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 5. Criptografar
            print("   🔒 Criptografando...")
            key_bytes = hashlib.sha256(master_key.encode()).digest()
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)
            
            encrypted_data = fernet.encrypt(serialized_data)
            
            # 6. Salvar com header personalizado
            secure_data = {
                'format': 'SecureTradingModel',
                'version': self.version,
                'encrypted_payload': encrypted_data
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(secure_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"   ✅ Modelo protegido salvo: {output_path}")
            print(f"   🔑 Hardware Lock: {'Ativo' if hardware_lock else 'Desativo'}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erro ao proteger modelo: {e}")
            return False
    
    def load_secure(self, 
                   secure_path: str, 
                   master_key: str,
                   validate_hardware: bool = True) -> Any:
        """
        Carrega modelo de arquivo seguro
        
        Args:
            secure_path: Caminho do arquivo .secure
            master_key: Chave mestra para descriptografia
            validate_hardware: Se deve validar hardware lock
        """
        try:
            print(f"🔓 Carregando modelo seguro: {os.path.basename(secure_path)}")
            
            # 1. Carregar dados
            with open(secure_path, 'rb') as f:
                secure_data = pickle.load(f)
            
            # 2. Validar formato
            if secure_data.get('format') != 'SecureTradingModel':
                raise ValueError("❌ Formato de arquivo inválido")
            
            # 3. Descriptografar
            print("   🔓 Descriptografando...")
            key_bytes = hashlib.sha256(master_key.encode()).digest()
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)
            
            decrypted_data = fernet.decrypt(secure_data['encrypted_payload'])
            model_data = pickle.loads(decrypted_data)
            
            # 4. Validar hardware (se necessário)
            if validate_hardware and model_data.get('hardware_fingerprint'):
                current_hw = HardwareFingerprint.generate()
                stored_hw = model_data['hardware_fingerprint']
                
                if current_hw != stored_hw:
                    raise ValueError(f"🚨 ACESSO NEGADO - Hardware não autorizado")
                    
                print("   ✅ Hardware autorizado")
            
            # 5. Desobfuscar pesos
            print("   🔄 Desobfuscando pesos...")
            state_dict = {}
            for key, tensor in model_data['state_dict'].items():
                if isinstance(tensor, torch.Tensor):
                    state_dict[key] = ModelObfuscator.deobfuscate_tensor(tensor, master_key)
                else:
                    state_dict[key] = tensor
            
            # 6. Criar objeto modelo (placeholder - precisa ser adaptado)
            print("   🔨 Reconstruindo modelo...")
            
            # NOTA: Aqui você precisaria criar uma instância do modelo
            # e carregar o state_dict. Isso depende da arquitetura específica.
            model_info = {
                'state_dict': state_dict,
                'model_type': model_data.get('model_type', 'PPO'),
                'created_at': model_data.get('created_at'),
                'additional_info': model_data.get('additional_info', {})
            }
            
            print(f"   ✅ Modelo carregado com sucesso")
            return model_info
            
        except Exception as e:
            print(f"   ❌ Erro ao carregar modelo seguro: {e}")
            raise


class ModelSecurityManager:
    """Manager principal para operações de segurança"""
    
    def __init__(self, master_key: str):
        self.master_key = master_key
        self.wrapper = SecureModelWrapper()
        
    def convert_checkpoint(self, 
                          input_path: str, 
                          output_path: Optional[str] = None,
                          hardware_lock: bool = True) -> bool:
        """
        Converte checkpoint normal para formato seguro
        
        Args:
            input_path: Caminho do checkpoint .zip normal
            output_path: Caminho de saída (opcional, auto-gera se None)
            hardware_lock: Ativar hardware lock
        """
        if not os.path.exists(input_path):
            print(f"❌ Arquivo não encontrado: {input_path}")
            return False
            
        if output_path is None:
            output_path = input_path.replace('.zip', '.secure')
            
        try:
            # Carregar modelo original
            print(f"📥 Carregando modelo original...")
            if 'RecurrentPPO' in input_path or 'Cherry' in input_path:
                model = RecurrentPPO.load(input_path)
            else:
                model = PPO.load(input_path)
            
            # Converter para formato seguro
            success = self.wrapper.save_secure(
                model=model,
                output_path=output_path,
                master_key=self.master_key,
                hardware_lock=hardware_lock
            )
            
            if success:
                file_size_mb = os.path.getsize(output_path) / (1024*1024)
                print(f"📊 Arquivo seguro criado: {file_size_mb:.2f} MB")
                
            return success
            
        except Exception as e:
            print(f"❌ Erro na conversão: {e}")
            return False
    
    def batch_convert(self, 
                     input_pattern: str,
                     output_dir: Optional[str] = None,
                     hardware_lock: bool = True) -> Dict[str, bool]:
        """
        Converte múltiplos checkpoints em batch
        
        Args:
            input_pattern: Padrão glob dos arquivos (ex: "models/Cherry/*.zip")
            output_dir: Diretório de saída (opcional)
            hardware_lock: Ativar hardware lock
        """
        import glob
        
        results = {}
        files = glob.glob(input_pattern)
        
        print(f"🔄 Convertendo {len(files)} modelos...")
        print("=" * 60)
        
        for i, file_path in enumerate(files, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{len(files)}] {filename}")
            
            if output_dir:
                output_path = os.path.join(output_dir, filename.replace('.zip', '.secure'))
            else:
                output_path = file_path.replace('.zip', '.secure')
                
            success = self.convert_checkpoint(file_path, output_path, hardware_lock)
            results[filename] = success
            
            print()
        
        # Resumo
        successful = sum(results.values())
        print("=" * 60)
        print(f"📊 RESUMO: {successful}/{len(files)} modelos convertidos com sucesso")
        
        if successful < len(files):
            print("❌ Falhas:")
            for filename, success in results.items():
                if not success:
                    print(f"   - {filename}")
                    
        return results
    
    def verify_secure_model(self, secure_path: str) -> bool:
        """Verifica se modelo seguro pode ser carregado"""
        try:
            model_info = self.wrapper.load_secure(secure_path, self.master_key)
            print(f"✅ Modelo válido: {secure_path}")
            print(f"   Tipo: {model_info.get('model_type', 'Unknown')}")
            print(f"   Criado: {time.ctime(model_info.get('created_at', 0))}")
            return True
        except Exception as e:
            print(f"❌ Modelo inválido: {e}")
            return False


# Funções de conveniência para uso direto
def protect_cherry_models(master_key: str = "cherry_trading_2025_v1"):
    """Protege todos os modelos Cherry existentes"""
    manager = ModelSecurityManager(master_key)
    
    pattern = "D:/Projeto/Otimizacao/treino_principal/models/Cherry/*.zip"
    return manager.batch_convert(pattern, hardware_lock=True)


def protect_single_model(model_path: str, master_key: str, hardware_lock: bool = True):
    """Protege um único modelo"""
    manager = ModelSecurityManager(master_key)
    return manager.convert_checkpoint(model_path, hardware_lock=hardware_lock)


if __name__ == "__main__":
    # Exemplo de uso
    print("🔒 Sistema de Proteção de Modelos Trading")
    print("=" * 50)
    
    # Demonstração
    hw_fingerprint = HardwareFingerprint.generate()
    print(f"Hardware Fingerprint desta máquina: {hw_fingerprint}")
    
    # Para testar, descomente:
    # protect_cherry_models()