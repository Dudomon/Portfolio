#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 SISTEMA DE LOGGING JSON PARA MONITORAMENTO EM TEMPO REAL

Sistema de logging otimizado que substitui CSV por JSON streaming
para permitir monitoramento em tempo real de gradientes e convergência.

Features:
- Logging JSON line-by-line (JSONL) para stream real-time
- Buffer em memória para performance
- Monitor de gradientes com alertas
- Análise de convergência automática
- WebSocket server para dashboard tempo real
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from collections import deque, defaultdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Instância global do logger
_global_logger = None

class RealTimeLogger:
    """
    🚀 LOGGER JSON STREAMING PARA MONITORAMENTO TEMPO REAL
    
    Substitui CSV por JSONL (JSON Lines) permitindo:
    - Leitura em tempo real sem travamento
    - Buffer circular para eficiência
    - Stream de dados para dashboards
    - Análise contínua de convergência
    """
    
    def __init__(self, 
                 base_path: str = "D:/Projeto/avaliacoes",
                 buffer_size: int = 2000,  # Buffer maior
                 flush_interval: float = 5.0,  # Flush menos frequente
                 cleanup_old_files: bool = True):  # Limpar arquivos antigos
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Buffer configuration
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Session ID único com PID + hash para múltiplas sessões simultâneas  
        import os
        import hashlib
        import random
        
        pid = os.getpid()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Hash único baseado em PID + timestamp + random para evitar colisões
        unique_hash = hashlib.md5(f"{pid}_{timestamp}_{random.randint(1000,9999)}".encode()).hexdigest()[:8]
        self.session_id = f"{timestamp}_{pid}_{unique_hash}"
        
        # Cleanup de arquivos antigos se solicitado (só do PID atual)
        if cleanup_old_files:
            self._cleanup_old_jsonl_files()
        
        # Buffers em memória para diferentes tipos de dados
        self.buffers = {
            'training': deque(maxlen=buffer_size),
            'gradients': deque(maxlen=buffer_size),
            'convergence': deque(maxlen=buffer_size),
            'rewards': deque(maxlen=buffer_size),
            'performance': deque(maxlen=buffer_size)
        }
        
        # File handles para cada stream
        self.file_handles = {}
        self._init_file_handles()
        
        # Thread para flush automático
        self.flush_thread = None
        self.running = False
        self.start_auto_flush()
        
        # Estatísticas em tempo real
        self.stats = defaultdict(list)
        self.last_flush = time.time()
        
        print(f"[FIRE] RealTimeLogger iniciado - Session: {self.session_id}")
        print(f"[FOLDER] Diretorio: {self.base_path}")
        
        # Registrar como instância global
        global _global_logger
        _global_logger = self
    
    def _cleanup_old_jsonl_files(self):
        """Remove APENAS arquivos JSONL antigos (não sessões simultâneas)"""
        try:
            # Só remove arquivos antigos (mais de 1 dia) ou da mesma sessão
            import os
            from datetime import datetime, timedelta
            
            jsonl_files = list(self.base_path.glob("*.jsonl"))
            deleted_count = 0
            current_time = datetime.now()
            current_pid = os.getpid()
            
            for file_path in jsonl_files:
                try:
                    # Parse filename para extrair timestamp e PID
                    filename = file_path.name
                    
                    # Skip arquivos de sessões ativas diferentes (mesmo PID = mesma sessão)
                    if f"_{current_pid}" in filename:
                        # Arquivo do mesmo PID - pode deletar
                        file_path.unlink()
                        deleted_count += 1
                    else:
                        # Arquivo de PID diferente - verificar idade
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if current_time - file_mtime > timedelta(days=1):
                            # Arquivo antigo (>1 dia) - pode deletar
                            file_path.unlink()
                            deleted_count += 1
                        # else: Arquivo recente de outro PID - preservar
                        
                except Exception as e:
                    print(f"[WARNING] Erro ao deletar {file_path}: {e}")
            
            if deleted_count > 0:
                print(f"[CLEANUP] {deleted_count} arquivos JSONL antigos removidos")
            else:
                print(f"[CLEANUP] Nenhum arquivo JSONL antigo encontrado")
                
        except Exception as e:
            print(f"[WARNING] Erro no cleanup: {e}")
    
    def _init_file_handles(self):
        """Inicializa arquivos JSONL para cada categoria de dados"""
        categories = ['training', 'gradients', 'convergence', 'rewards', 'performance']
        
        for category in categories:
            filename = f"{category}_{self.session_id}.jsonl"
            filepath = self.base_path / filename
            
            # Abrir arquivo em modo append com buffer mínimo
            self.file_handles[category] = open(filepath, 'a', buffering=1)
            
            # Escrever header com metadados
            header = {
                'type': 'header',
                'category': category,
                'session_id': self.session_id,
                'start_time': datetime.now().isoformat(),
                'version': '1.0'
            }
            self.file_handles[category].write(json.dumps(header) + '\n')
            self.file_handles[category].flush()
    
    def log_training_step(self, step: int, metrics: Dict[str, Any]):
        """Log dados de treinamento por step"""
        entry = {
            'type': 'training_step',
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'session_id': self.session_id,
            **metrics
        }
        
        self.buffers['training'].append(entry)
        self._update_stats('training', metrics)
    
    def log_gradient_info(self, step: int, gradient_data: Dict[str, Any]):
        """Log informações de gradientes"""
        entry = {
            'type': 'gradient_info',
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'session_id': self.session_id,
            **gradient_data
        }
        
        self.buffers['gradients'].append(entry)
        self._update_stats('gradients', gradient_data)
        
        # Análise automática de gradientes
        self._analyze_gradient_health(gradient_data)
    
    def log_convergence_metrics(self, step: int, convergence_data: Dict[str, Any]):
        """Log métricas de convergência"""
        entry = {
            'type': 'convergence_metrics',
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'session_id': self.session_id,
            **convergence_data
        }
        
        self.buffers['convergence'].append(entry)
        self._update_stats('convergence', convergence_data)
        
        # Análise automática de convergência
        self._analyze_convergence_health(convergence_data)
    
    def log_reward_info(self, step: int, reward_data: Dict[str, Any]):
        """Log informações de reward"""
        entry = {
            'type': 'reward_info',
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'session_id': self.session_id,
            **reward_data
        }
        
        self.buffers['rewards'].append(entry)
        self._update_stats('rewards', reward_data)
    
    def log_performance_metrics(self, step: int, performance_data: Dict[str, Any]):
        """Log métricas de performance"""
        entry = {
            'type': 'performance_metrics',
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'session_id': self.session_id,
            **performance_data
        }
        
        self.buffers['performance'].append(entry)
        self._update_stats('performance', performance_data)
    
    def log_performance_metric(self, step: int, metric_type: str, metric_data: Dict[str, Any]):
        """Log métricas específicas por tipo (actor, critic, rewards)"""
        entry = {
            'type': f'{metric_type}_metrics',
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'session_id': self.session_id,
            'metric_type': metric_type,
            **metric_data
        }
        
        # Usar buffer apropriado baseado no tipo
        buffer_name = 'performance'  # Default
        if metric_type == 'rewards':
            buffer_name = 'rewards'
        elif metric_type in ['actor', 'critic']:
            buffer_name = 'training'  # Métricas de treinamento
            
        self.buffers[buffer_name].append(entry)
        self._update_stats(buffer_name, metric_data)
    
    def _update_stats(self, category: str, data: Dict[str, Any]):
        """Atualiza estatísticas em tempo real"""
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.stats[f"{category}_{key}"].append(value)
                
                # Manter apenas últimos 1000 valores
                if len(self.stats[f"{category}_{key}"]) > 1000:
                    self.stats[f"{category}_{key}"].pop(0)
    
    def _analyze_gradient_health(self, gradient_data: Dict[str, Any]):
        """Análise automática de saúde dos gradientes"""
        if 'grad_norm' in gradient_data:
            grad_norm = gradient_data['grad_norm']
            
            # Alertas automáticos - mais tolerantes
            if grad_norm > 10.0:
                self._log_alert("HIGH_GRADIENT", f"Gradient norm muito alto: {grad_norm:.4f}")
            elif grad_norm > 0 and grad_norm < 1e-10:  # Só alertar se > 0 mas muito pequeno
                self._log_alert("VANISHING_GRADIENT", f"Gradient norm muito baixo: {grad_norm:.10f}")
            # Não alertar para grad_norm == 0 (dados não disponíveis)
        
        if 'grad_zeros_ratio' in gradient_data:
            zeros_ratio = gradient_data['grad_zeros_ratio']
            if zeros_ratio > 0.7:  # Mais tolerante: 70% em vez de 50%
                self._log_alert("GRADIENT_ZEROS", f"Muitos zeros nos gradientes: {zeros_ratio*100:.1f}%")
    
    def _analyze_convergence_health(self, convergence_data: Dict[str, Any]):
        """Análise automática de convergência"""
        if 'loss' in convergence_data and len(self.stats['convergence_loss']) > 10:
            recent_losses = self.stats['convergence_loss'][-10:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            if loss_trend > 0.01:  # Loss aumentando
                self._log_alert("DIVERGENCE", f"Loss divergindo: tendência +{loss_trend:.6f}")
            elif abs(loss_trend) < 1e-6:  # Loss estagnado
                self._log_alert("STAGNATION", f"Loss estagnado: tendência {loss_trend:.8f}")
    
    def _log_alert(self, alert_type: str, message: str):
        """Log alertas importantes"""
        alert = {
            'type': 'alert',
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'message': message,
            'session_id': self.session_id
        }
        
        # Alert vai para todos os streams
        for category in self.buffers:
            self.buffers[category].append(alert)
        
        print(f"[ALERT] [{alert_type}]: {message}")
    
    def start_auto_flush(self):
        """Inicia thread para flush automático dos buffers"""
        if self.flush_thread is None or not self.flush_thread.is_alive():
            self.running = True
            self.flush_thread = threading.Thread(target=self._auto_flush_loop, daemon=True)
            self.flush_thread.start()
    
    def _auto_flush_loop(self):
        """Loop principal para flush automático"""
        while self.running:
            try:
                time.sleep(self.flush_interval)
                self.flush_all_buffers()
            except Exception as e:
                print(f"[WARNING] Erro no auto-flush: {e}")
    
    def flush_all_buffers(self):
        """Flush todos os buffers para disco"""
        flushed_count = 0
        
        for category, buffer in self.buffers.items():
            if buffer and category in self.file_handles:
                # Escrever todos os itens do buffer
                while buffer:
                    entry = buffer.popleft()
                    self.file_handles[category].write(json.dumps(entry) + '\n')
                    flushed_count += 1
                
                # Força flush do sistema
                self.file_handles[category].flush()
        
        if flushed_count > 0:
            self.last_flush = time.time()
            # Flush silencioso - debug removido
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas em tempo real"""
        stats_summary = {}
        
        for key, values in self.stats.items():
            if values:
                stats_summary[key] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'stats': stats_summary,
            'buffer_sizes': {k: len(v) for k, v in self.buffers.items()},
            'last_flush': self.last_flush
        }
    
    def close(self):
        """Fechar logger e arquivos"""
        print(f"[LOCK] Fechando RealTimeLogger - Session: {self.session_id}")
        
        # Parar thread de flush
        self.running = False
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5.0)
        
        # Flush final
        self.flush_all_buffers()
        
        # Fechar arquivos
        for handle in self.file_handles.values():
            if not handle.closed:
                handle.close()
        
        print("[CHECK] RealTimeLogger fechado com sucesso")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LogReader:
    """
    📊 LEITOR DE LOGS EM TEMPO REAL
    
    Permite ler logs JSONL em tempo real para dashboards e monitoramento.
    """
    
    def __init__(self, base_path: str = "D:/Projeto/avaliacoes"):
        self.base_path = Path(base_path)
    
    def get_latest_session(self) -> Optional[str]:
        """Encontra a sessão mais recente"""
        sessions = set()
        
        for file in self.base_path.glob("*.jsonl"):
            # Extract session_id from filename: category_YYYYMMDD_HHMMSS.jsonl
            parts = file.stem.split('_')
            if len(parts) >= 3:
                session_id = '_'.join(parts[-2:])  # YYYYMMDD_HHMMSS
                sessions.add(session_id)
        
        return max(sessions) if sessions else None
    
    def read_stream(self, category: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lê stream de dados de uma categoria"""
        if session_id is None:
            session_id = self.get_latest_session()
            if session_id is None:
                return []
        
        filename = f"{category}_{session_id}.jsonl"
        filepath = self.base_path / filename
        
        if not filepath.exists():
            return []
        
        entries = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            print(f"[WARNING] Erro ao ler {filepath}: {e}")
        
        return entries
    
    def tail_stream(self, category: str, n: int = 100, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lê últimas N entradas de um stream"""
        entries = self.read_stream(category, session_id)
        return entries[-n:] if entries else []
    
    def get_alerts(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtém todos os alertas da sessão"""
        alerts = []
        categories = ['training', 'gradients', 'convergence', 'rewards', 'performance']
        
        for category in categories:
            entries = self.read_stream(category, session_id)
            alerts.extend([e for e in entries if e.get('type') == 'alert'])
        
        # Ordenar por timestamp
        alerts.sort(key=lambda x: x.get('timestamp', ''))
        return alerts


def create_real_time_logger(**kwargs) -> RealTimeLogger:
    """Factory function para criar logger"""
    return RealTimeLogger(**kwargs)


def create_log_reader(**kwargs) -> LogReader:
    """Factory function para criar reader"""
    return LogReader(**kwargs)


# Teste rápido
if __name__ == "__main__":
    # Teste básico do logger
    with create_real_time_logger() as logger:
        print("[TEST] Testando RealTimeLogger...")
        
        # Simular dados de treinamento
        for step in range(10):
            logger.log_training_step(step, {
                'loss': np.random.uniform(0.1, 2.0),
                'learning_rate': 2.5e-4,
                'episode_reward': np.random.uniform(-100, 100)
            })
            
            logger.log_gradient_info(step, {
                'grad_norm': np.random.uniform(0.01, 5.0),
                'grad_zeros_ratio': np.random.uniform(0.0, 0.3)
            })
            
            time.sleep(0.1)

def get_logger():
    """Retorna a instância global do logger"""
    return _global_logger

def create_logger(*args, **kwargs):
    """Cria e retorna uma nova instância do logger"""
    return RealTimeLogger(*args, **kwargs)