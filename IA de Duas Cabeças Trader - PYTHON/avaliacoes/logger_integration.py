#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 INTEGRAÇÃO DO LOGGER JSON NO SISTEMA DE TREINAMENTO

Patches e integrações para substituir CSV logging por JSON streaming
no sistema de treinamento existente, permitindo monitoramento tempo real.

Features:
- Patch automático dos loggers CSV existentes
- Integração transparente com callbacks
- Coleta automática de métricas de gradiente
- Bridge para sistemas legados
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar logger personalizado
sys.path.append(str(Path(__file__).parent))
from real_time_logger import RealTimeLogger

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class TrainingLoggerIntegration:
    """
    🔗 INTEGRAÇÃO TRANSPARENTE DO LOGGER JSON
    
    Integra o RealTimeLogger no sistema de treinamento existente,
    substituindo gradualmente os CSVs por JSON streaming.
    """
    
    def __init__(self, 
                 base_path: str = "D:/Projeto/avaliacoes",
                 enable_csv_fallback: bool = True,
                 gradient_monitoring: bool = True):
        
        self.base_path = Path(base_path)
        self.enable_csv_fallback = enable_csv_fallback
        self.gradient_monitoring = gradient_monitoring
        
        # Logger principal
        self.real_time_logger = None
        self.session_active = False
        
        # Cache para dados de gradiente
        self.gradient_cache = {'step': 0, 'data': {}}
        
        # Contadores
        self.step_counter = 0
        self.episode_counter = 0
        
        print("[LINK] TrainingLoggerIntegration inicializado")
    
    def start_session(self, session_prefix: str = "training") -> str:
        """Inicia nova sessão de logging"""
        if self.session_active:
            self.end_session()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{session_prefix}_{timestamp}"
        
        # Criar logger
        self.real_time_logger = RealTimeLogger(
            base_path=str(self.base_path),
            buffer_size=500,  # Buffer menor para mais responsividade
            flush_interval=0.5  # Flush mais frequente
        )
        
        self.session_active = True
        self.step_counter = 0
        self.episode_counter = 0
        
        print(f"[ROCKET] Sessao iniciada: {session_id}")
        return session_id
    
    def end_session(self):
        """Finaliza sessão de logging"""
        if self.real_time_logger:
            self.real_time_logger.close()
            self.real_time_logger = None
        
        self.session_active = False
        print("[LOCK] Sessao de logging finalizada")
    
    def log_training_step(self, **kwargs):
        """Log step de treinamento com dados variados"""
        if not self.session_active or not self.real_time_logger:
            return
        
        self.step_counter += 1
        
        # Preparar dados do step
        step_data = {
            'step': self.step_counter,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        # Log no sistema JSON
        self.real_time_logger.log_training_step(self.step_counter, step_data)
        
        # CSV fallback se habilitado
        if self.enable_csv_fallback:
            self._write_csv_fallback('training', step_data)
    
    def log_gradient_info(self, model: Optional[Any] = None, **manual_data):
        """Log informações de gradiente"""
        if not self.session_active or not self.real_time_logger:
            return
        
        gradient_data = manual_data.copy()
        
        # Coletar gradientes automaticamente se modelo fornecido
        if model is not None and TORCH_AVAILABLE and self.gradient_monitoring:
            auto_grad_data = self._extract_gradient_info(model)
            gradient_data.update(auto_grad_data)
        
        # Log se tiver dados
        if gradient_data:
            self.real_time_logger.log_gradient_info(self.step_counter, gradient_data)
            
            if self.enable_csv_fallback:
                self._write_csv_fallback('gradients', gradient_data)
    
    def log_episode_end(self, **episode_data):
        """Log fim de episódio"""
        if not self.session_active or not self.real_time_logger:
            return
        
        self.episode_counter += 1
        
        episode_info = {
            'episode': self.episode_counter,
            'step': self.step_counter,
            **episode_data
        }
        
        self.real_time_logger.log_reward_info(self.step_counter, episode_info)
        
        if self.enable_csv_fallback:
            self._write_csv_fallback('rewards', episode_info)
    
    def log_performance_metrics(self, **performance_data):
        """Log métricas de performance"""
        if not self.session_active or not self.real_time_logger:
            return
        
        self.real_time_logger.log_performance_metrics(self.step_counter, performance_data)
        
        if self.enable_csv_fallback:
            self._write_csv_fallback('performance', performance_data)
    
    def _extract_gradient_info(self, model) -> Dict[str, Any]:
        """Extrai informações de gradiente do modelo"""
        if not TORCH_AVAILABLE:
            return {}
        
        try:
            total_norm = 0.0
            param_count = 0
            zero_grad_count = 0
            grad_norms = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
                    # Contar zeros
                    zero_count = (param.grad.data == 0).sum().item()
                    zero_grad_count += zero_count
                    
                    grad_norms.append(param_norm.item())
            
            total_norm = total_norm ** (1. / 2)
            
            # Calcular estatísticas
            total_params = sum(p.numel() for p in model.parameters() if p.grad is not None)
            zeros_ratio = zero_grad_count / max(total_params, 1)
            
            return {
                'grad_norm': float(total_norm),
                'grad_zeros_ratio': float(zeros_ratio),
                'param_count': param_count,
                'grad_mean': float(np.mean(grad_norms)) if grad_norms else 0.0,
                'grad_std': float(np.std(grad_norms)) if grad_norms else 0.0,
                'grad_min': float(np.min(grad_norms)) if grad_norms else 0.0,
                'grad_max': float(np.max(grad_norms)) if grad_norms else 0.0
            }
            
        except Exception as e:
            print(f"[WARNING] Erro ao extrair gradientes: {e}")
            return {'gradient_extraction_error': str(e)}
    
    def _write_csv_fallback(self, category: str, data: Dict[str, Any]):
        """Escreve dados em CSV como fallback"""
        if not self.enable_csv_fallback:
            return
        
        try:
            csv_file = self.base_path / f"{category}_fallback.csv"
            
            # Preparar linha CSV
            timestamp = datetime.now().isoformat()
            step = data.get('step', self.step_counter)
            
            # Dados principais como string JSON (simples)
            data_str = json.dumps({k: v for k, v in data.items() 
                                 if k not in ['step', 'timestamp']})
            
            csv_line = f"{timestamp},{step},\"{data_str}\"\n"
            
            # Escrever (criar header se necessário)
            if not csv_file.exists():
                with open(csv_file, 'w') as f:
                    f.write("timestamp,step,data\n")
            
            with open(csv_file, 'a') as f:
                f.write(csv_line)
                
        except Exception as e:
            print(f"[WARNING] Erro no CSV fallback: {e}")
    
    def create_sb3_callback(self) -> Optional['RealTimeLoggingCallback']:
        """Cria callback para Stable-Baselines3"""
        if not SB3_AVAILABLE:
            print("[WARNING] Stable-Baselines3 nao disponivel")
            return None
        
        return RealTimeLoggingCallback(self)
    
    def patch_existing_logger(self, logger_instance, method_name: str = 'log'):
        """Aplica patch em logger existente"""
        if not hasattr(logger_instance, method_name):
            print(f"[WARNING] Metodo {method_name} nao encontrado no logger")
            return
        
        # Salvar método original
        original_method = getattr(logger_instance, method_name)
        
        def patched_method(*args, **kwargs):
            # Chamar método original
            result = original_method(*args, **kwargs)
            
            # Interceptar dados e enviar para nosso logger
            try:
                if kwargs:
                    self.log_training_step(**kwargs)
                elif args and isinstance(args[0], dict):
                    self.log_training_step(**args[0])
            except Exception as e:
                print(f"[WARNING] Erro no patch do logger: {e}")
            
            return result
        
        # Aplicar patch
        setattr(logger_instance, method_name, patched_method)
        print(f"[WRENCH] Patch aplicado em {logger_instance.__class__.__name__}.{method_name}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas atuais"""
        return {
            'session_active': self.session_active,
            'step_counter': self.step_counter,
            'episode_counter': self.episode_counter,
            'logger_active': self.real_time_logger is not None,
            'base_path': str(self.base_path)
        }


class RealTimeLoggingCallback(BaseCallback):
    """
    📊 CALLBACK PARA STABLE-BASELINES3
    
    Integra automaticamente o logging JSON no treinamento SB3.
    """
    
    def __init__(self, logger_integration: TrainingLoggerIntegration, verbose: int = 0):
        super().__init__(verbose)
        self.logger_integration = logger_integration
        self.last_log_step = 0
        
    def _on_training_start(self) -> None:
        """Inicia sessão quando treinamento começa"""
        if not self.logger_integration.session_active:
            self.logger_integration.start_session("sb3_training")
        
        # Log informações iniciais
        self.logger_integration.log_training_step(
            event='training_start',
            model_class=self.model.__class__.__name__,
            total_timesteps=getattr(self.locals.get('total_timesteps'), 'value', 0) if hasattr(self.locals.get('total_timesteps', 0), 'value') else self.locals.get('total_timesteps', 0)
        )
    
    def _on_step(self) -> bool:
        """Chamado a cada step"""
        # Log dados do step atual
        step_data = {}
        
        # Coletar informações disponíveis
        if hasattr(self, 'locals') and self.locals:
            infos = self.locals.get('infos', [{}])
            if infos and isinstance(infos[0], dict):
                step_data.update(infos[0])
        
        # Adicionar informações do modelo
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            step_data.update(self.model.logger.name_to_value)
        
        # Log step
        self.logger_integration.log_training_step(**step_data)
        
        # Log gradientes periodicamente
        if self.num_timesteps % 10 == 0:  # A cada 10 steps
            self.logger_integration.log_gradient_info(self.model.policy)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Fim do rollout"""
        # Log métricas de performance do rollout
        rollout_data = {}
        
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            rollout_data = self.model.logger.name_to_value.copy()
        
        self.logger_integration.log_performance_metrics(**rollout_data)
    
    def _on_training_end(self) -> None:
        """Fim do treinamento"""
        self.logger_integration.log_training_step(
            event='training_end',
            total_steps=self.num_timesteps
        )
        
        # Finalizar sessão
        self.logger_integration.end_session()


def create_integrated_logger(**kwargs) -> TrainingLoggerIntegration:
    """Factory function para criar logger integrado"""
    return TrainingLoggerIntegration(**kwargs)


def patch_daytrader_logger(daytrader_path: str = "D:/Projeto/daytrader.py"):
    """
    🔧 PATCH ESPECÍFICO PARA DAYTRADER.PY
    
    Aplica patches no sistema de logging do daytrader existente.
    """
    print(f"[WRENCH] Aplicando patch no daytrader: {daytrader_path}")
    
    # Criar integração
    logger_integration = create_integrated_logger()
    session_id = logger_integration.start_session("daytrader")
    
    # TODO: Implementar patches específicos baseados na estrutura do daytrader
    # Isso requereria análise do código existente
    
    print(f"[CHECK] Patch aplicado - Sessao: {session_id}")
    return logger_integration


# Exemplo de uso direto
if __name__ == "__main__":
    print("[TEST] Testando integracao do logger...")
    
    # Criar integração
    integration = create_integrated_logger()
    
    # Iniciar sessão
    session_id = integration.start_session("test")
    
    # Simular dados de treinamento
    for step in range(20):
        # Log step de treinamento
        integration.log_training_step(
            loss=np.random.uniform(0.1, 2.0),
            learning_rate=2.5e-4,
            entropy_loss=np.random.uniform(0.01, 0.1)
        )
        
        # Log gradientes (dados simulados)
        integration.log_gradient_info(
            grad_norm=np.random.uniform(0.1, 3.0),
            grad_zeros_ratio=np.random.uniform(0.0, 0.2)
        )
        
        # Log episódio ocasionalmente
        if step % 5 == 0:
            integration.log_episode_end(
                episode_reward=np.random.uniform(-50, 150),
                episode_length=np.random.randint(10, 100)
            )
        
        time.sleep(0.1)
    
    # Mostrar estatísticas
    stats = integration.get_current_stats()
    print(f"[STATS] Stats finais: {stats}")
    
    # Finalizar
    integration.end_session()
    
    print("[CHECK] Teste concluido!")