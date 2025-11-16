#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 DAYPROFILE - SISTEMA DE PROFILING EM TEMPO REAL
Sistema completo de monitoramento de performance para identificar gargalos no treinamento
"""

import sys
import os
import time
import threading
import psutil
import gc
import traceback
from datetime import datetime
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json
import csv

# Profiling libraries (com fallbacks)
import cProfile
import pstats
import io

# Optional profiling libraries (com fallbacks)
try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False
    print("⚠️ line_profiler não disponível - profiling de linha desabilitado")

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("⚠️ memory_profiler não disponível - profiling de memória simplificado")

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("⚠️ py3nvml não disponível - monitoramento GPU simplificado")

# ML libraries
import torch
import numpy as np
import pandas as pd

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io as sys_io
    sys.stdout = sys_io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = sys_io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

@dataclass
class ProfileMetrics:
    """Métricas de profiling"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float
    gpu_utilization: float
    function_times: Dict[str, float]
    memory_usage: Dict[str, float]
    thread_count: int
    io_stats: Dict[str, Any]

class RealTimeProfiler:
    """🔍 Sistema de profiling em tempo real completo"""
    
    def __init__(self, output_dir="profiling_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuração
        self.monitoring_active = False
        self.profile_interval = 0.1  # 100ms
        self.detailed_interval = 1.0  # 1s para análises detalhadas
        
        # Buffers de dados
        self.metrics_buffer = deque(maxlen=10000)  # ~16 minutos de dados
        self.function_stats = defaultdict(list)
        self.memory_timeline = deque(maxlen=1000)
        self.gpu_timeline = deque(maxlen=1000)
        
        # Profilers
        self.cpu_profiler = cProfile.Profile()
        self.line_profiler = LineProfiler() if LINE_PROFILER_AVAILABLE else None
        self.memory_profiler = memory_profiler if MEMORY_PROFILER_AVAILABLE else None
        
        # Threading
        self.monitor_thread = None
        self.detailed_thread = None
        self.stop_event = threading.Event()
        
        # GPU monitoring
        self.gpu_available = self._init_gpu_monitoring()
        
        # Arquivos de output
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = f"{output_dir}/realtime_metrics_{self.timestamp}.csv"
        self.detailed_log = f"{output_dir}/detailed_analysis_{self.timestamp}.log"
        
        # Inicializar CSV
        self._init_csv_file()
        
        print(f"🔍 RealTimeProfiler inicializado")
        print(f"📊 Output: {output_dir}")
        print(f"🎯 GPU disponível: {self.gpu_available}")
        
    def _init_gpu_monitoring(self):
        """Inicializar monitoramento GPU"""
        if not NVML_AVAILABLE:
            return False
        
        try:
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            if device_count > 0:
                self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
                return True
        except:
            pass
        return False
    
    def _init_csv_file(self):
        """Inicializar arquivo CSV com headers"""
        headers = [
            'timestamp', 'elapsed_time', 'cpu_percent', 'memory_mb', 'memory_percent',
            'gpu_memory_mb', 'gpu_utilization', 'thread_count', 'gc_count',
            'torch_allocated_mb', 'torch_cached_mb', 'torch_reserved_mb',
            'io_read_mb', 'io_write_mb', 'network_sent_mb', 'network_recv_mb',
            'top_function', 'top_function_time', 'memory_growth_rate',
            'cpu_load_1m', 'cpu_load_5m', 'swap_usage_mb'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def start_monitoring(self):
        """🚀 Iniciar monitoramento em tempo real"""
        if self.monitoring_active:
            print("⚠️ Monitoramento já ativo")
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        self.start_time = time.time()
        
        # Thread principal de monitoramento (alta frequência)
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Thread de análise detalhada (baixa frequência)
        self.detailed_thread = threading.Thread(target=self._detailed_analysis_loop, daemon=True)
        self.detailed_thread.start()
        
        # Iniciar profiling CPU
        self.cpu_profiler.enable()
        
        print("🔍 Monitoramento em tempo real INICIADO")
        print(f"📊 Métricas sendo salvas em: {self.csv_file}")
        print(f"📋 Análise detalhada em: {self.detailed_log}")
    
    def stop_monitoring(self):
        """🛑 Parar monitoramento"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        # Parar profiling CPU
        self.cpu_profiler.disable()
        
        # Aguardar threads
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        if self.detailed_thread:
            self.detailed_thread.join(timeout=2.0)
        
        # Gerar relatório final
        self._generate_final_report()
        
        print("🛑 Monitoramento PARADO")
        print(f"📊 Relatório final gerado")
    
    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while not self.stop_event.wait(self.profile_interval):
            try:
                metrics = self._collect_metrics()
                self.metrics_buffer.append(metrics)
                self._save_metrics_to_csv(metrics)
                
                # Print em tempo real (a cada 10 coletas)
                if len(self.metrics_buffer) % 10 == 0:
                    self._print_realtime_status(metrics)
                    
            except Exception as e:
                print(f"❌ Erro no monitor loop: {e}")
    
    def _detailed_analysis_loop(self):
        """Loop de análise detalhada"""
        while not self.stop_event.wait(self.detailed_interval):
            try:
                self._analyze_performance_trends()
                self._detect_bottlenecks()
                self._analyze_memory_leaks()
                
            except Exception as e:
                print(f"❌ Erro na análise detalhada: {e}")
    
    def _collect_metrics(self):
        """📊 Coletar todas as métricas"""
        current_time = time.time()
        
        # Sistema
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        # Threads
        thread_count = process.num_threads()
        
        # I/O
        try:
            io_counters = process.io_counters()
            io_read_mb = io_counters.read_bytes / 1024 / 1024
            io_write_mb = io_counters.write_bytes / 1024 / 1024
        except:
            io_read_mb = io_write_mb = 0
        
        # Network
        try:
            net_io = psutil.net_io_counters()
            network_sent_mb = net_io.bytes_sent / 1024 / 1024
            network_recv_mb = net_io.bytes_recv / 1024 / 1024
        except:
            network_sent_mb = network_recv_mb = 0
        
        # GPU
        gpu_memory_mb = gpu_utilization = 0
        if self.gpu_available and NVML_AVAILABLE:
            try:
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_mb = mem_info.used / 1024 / 1024
                
                util_info = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_utilization = util_info.gpu
            except:
                pass
        
        # PyTorch memory
        torch_allocated_mb = torch_cached_mb = torch_reserved_mb = 0
        if torch.cuda.is_available():
            try:
                torch_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                torch_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
                torch_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
            except:
                pass
        
        # Garbage collection
        gc_count = len(gc.get_objects())
        
        # CPU load
        try:
            load_avg = os.getloadavg()
            cpu_load_1m = load_avg[0]
            cpu_load_5m = load_avg[1]
        except:
            cpu_load_1m = cpu_load_5m = 0
        
        # Swap
        try:
            swap_info = psutil.swap_memory()
            swap_usage_mb = swap_info.used / 1024 / 1024
        except:
            swap_usage_mb = 0
        
        return ProfileMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            function_times={},
            memory_usage={},
            thread_count=thread_count,
            io_stats={
                'read_mb': io_read_mb,
                'write_mb': io_write_mb,
                'network_sent_mb': network_sent_mb,
                'network_recv_mb': network_recv_mb,
                'memory_percent': memory_percent,
                'gc_count': gc_count,
                'torch_allocated_mb': torch_allocated_mb,
                'torch_cached_mb': torch_cached_mb,
                'torch_reserved_mb': torch_reserved_mb,
                'cpu_load_1m': cpu_load_1m,
                'cpu_load_5m': cpu_load_5m,
                'swap_usage_mb': swap_usage_mb
            }
        )
    
    def _save_metrics_to_csv(self, metrics: ProfileMetrics):
        """💾 Salvar métricas no CSV"""
        elapsed_time = metrics.timestamp - self.start_time
        io_stats = metrics.io_stats
        
        # Calcular taxa de crescimento de memória
        memory_growth_rate = 0
        if len(self.memory_timeline) > 1:
            recent_memory = [m.memory_mb for m in list(self.memory_timeline)[-10:]]
            if len(recent_memory) >= 2:
                memory_growth_rate = (recent_memory[-1] - recent_memory[0]) / len(recent_memory)
        
        self.memory_timeline.append(metrics)
        
        # Top function (placeholder - será implementado com profiling detalhado)
        top_function = "N/A"
        top_function_time = 0
        
        row = [
            metrics.timestamp, elapsed_time, metrics.cpu_percent, metrics.memory_mb,
            io_stats['memory_percent'], metrics.gpu_memory_mb, metrics.gpu_utilization,
            metrics.thread_count, io_stats['gc_count'], io_stats['torch_allocated_mb'],
            io_stats['torch_cached_mb'], io_stats['torch_reserved_mb'],
            io_stats['read_mb'], io_stats['write_mb'], io_stats['network_sent_mb'],
            io_stats['network_recv_mb'], top_function, top_function_time,
            memory_growth_rate, io_stats['cpu_load_1m'], io_stats['cpu_load_5m'],
            io_stats['swap_usage_mb']
        ]
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def _print_realtime_status(self, metrics: ProfileMetrics):
        """📺 Print status em tempo real"""
        elapsed = metrics.timestamp - self.start_time
        io_stats = metrics.io_stats
        
        # Limpar tela (opcional)
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"\n🔍 PROFILING EM TEMPO REAL - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        print(f"⏱️  Tempo decorrido: {elapsed:.1f}s")
        print(f"🖥️  CPU: {metrics.cpu_percent:.1f}% | Threads: {metrics.thread_count}")
        print(f"💾 RAM: {metrics.memory_mb:.1f}MB ({io_stats['memory_percent']:.1f}%)")
        print(f"🎮 GPU: {metrics.gpu_utilization:.1f}% | VRAM: {metrics.gpu_memory_mb:.1f}MB")
        print(f"🔥 PyTorch: Alloc={io_stats['torch_allocated_mb']:.1f}MB | Cache={io_stats['torch_cached_mb']:.1f}MB")
        print(f"💿 I/O: R={io_stats['read_mb']:.1f}MB | W={io_stats['write_mb']:.1f}MB")
        print(f"🌐 Net: S={io_stats['network_sent_mb']:.1f}MB | R={io_stats['network_recv_mb']:.1f}MB")
        print(f"🗑️  GC Objects: {io_stats['gc_count']:,}")
        
        # Alertas
        if metrics.cpu_percent > 90:
            print("⚠️  ALERTA: CPU usage alto!")
        if io_stats['memory_percent'] > 85:
            print("⚠️  ALERTA: Memória alta!")
        if metrics.gpu_memory_mb > 8000:  # 8GB
            print("⚠️  ALERTA: VRAM alta!")
        
        print("=" * 80)
    
    def _analyze_performance_trends(self):
        """📈 Analisar tendências de performance"""
        if len(self.metrics_buffer) < 10:
            return
        
        recent_metrics = list(self.metrics_buffer)[-60:]  # Último minuto
        
        # Tendências
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_mb for m in recent_metrics])
        gpu_trend = self._calculate_trend([m.gpu_utilization for m in recent_metrics])
        
        # Log tendências
        with open(self.detailed_log, 'a', encoding='utf-8') as f:
            f.write(f"\n[{datetime.now()}] ANÁLISE DE TENDÊNCIAS:\n")
            f.write(f"CPU Trend: {cpu_trend:.3f}\n")
            f.write(f"Memory Trend: {memory_trend:.3f}\n")
            f.write(f"GPU Trend: {gpu_trend:.3f}\n")
    
    def _detect_bottlenecks(self):
        """🔍 Detectar gargalos"""
        if len(self.metrics_buffer) < 5:
            return
        
        recent = list(self.metrics_buffer)[-5:]
        avg_cpu = np.mean([m.cpu_percent for m in recent])
        avg_memory = np.mean([m.memory_mb for m in recent])
        avg_gpu = np.mean([m.gpu_utilization for m in recent])
        
        bottlenecks = []
        
        if avg_cpu > 85:
            bottlenecks.append("CPU")
        if avg_memory > 8000:  # 8GB
            bottlenecks.append("MEMORY")
        if avg_gpu > 90:
            bottlenecks.append("GPU")
        
        if bottlenecks:
            with open(self.detailed_log, 'a', encoding='utf-8') as f:
                f.write(f"\n[{datetime.now()}] GARGALOS DETECTADOS: {', '.join(bottlenecks)}\n")
    
    def _analyze_memory_leaks(self):
        """🔍 Analisar vazamentos de memória"""
        if len(self.memory_timeline) < 20:
            return
        
        recent_memory = [m.memory_mb for m in list(self.memory_timeline)[-20:]]
        memory_growth = recent_memory[-1] - recent_memory[0]
        
        if memory_growth > 100:  # 100MB growth
            with open(self.detailed_log, 'a', encoding='utf-8') as f:
                f.write(f"\n[{datetime.now()}] POSSÍVEL VAZAMENTO: +{memory_growth:.1f}MB em 20 amostras\n")
    
    def _calculate_trend(self, values):
        """Calcular tendência linear"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.std(y) == 0:
            return 0.0
        
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _generate_final_report(self):
        """📋 Gerar relatório final"""
        if not self.metrics_buffer:
            return
        
        report_file = f"{self.output_dir}/final_report_{self.timestamp}.txt"
        
        # Estatísticas
        all_metrics = list(self.metrics_buffer)
        cpu_values = [m.cpu_percent for m in all_metrics]
        memory_values = [m.memory_mb for m in all_metrics]
        gpu_values = [m.gpu_utilization for m in all_metrics]
        
        total_time = all_metrics[-1].timestamp - all_metrics[0].timestamp
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"🔍 RELATÓRIO FINAL DE PROFILING\n")
            f.write(f"Gerado em: {datetime.now()}\n")
            f.write(f"=" * 60 + "\n\n")
            
            f.write(f"⏱️ DURAÇÃO TOTAL: {total_time:.1f} segundos\n")
            f.write(f"📊 AMOSTRAS COLETADAS: {len(all_metrics):,}\n\n")
            
            f.write(f"🖥️ CPU USAGE:\n")
            f.write(f"   Média: {np.mean(cpu_values):.1f}%\n")
            f.write(f"   Máximo: {np.max(cpu_values):.1f}%\n")
            f.write(f"   Mínimo: {np.min(cpu_values):.1f}%\n")
            f.write(f"   Desvio: {np.std(cpu_values):.1f}%\n\n")
            
            f.write(f"💾 MEMORY USAGE:\n")
            f.write(f"   Média: {np.mean(memory_values):.1f}MB\n")
            f.write(f"   Máximo: {np.max(memory_values):.1f}MB\n")
            f.write(f"   Mínimo: {np.min(memory_values):.1f}MB\n")
            f.write(f"   Crescimento: {memory_values[-1] - memory_values[0]:.1f}MB\n\n")
            
            if self.gpu_available:
                f.write(f"🎮 GPU USAGE:\n")
                f.write(f"   Média: {np.mean(gpu_values):.1f}%\n")
                f.write(f"   Máximo: {np.max(gpu_values):.1f}%\n")
                f.write(f"   Mínimo: {np.min(gpu_values):.1f}%\n\n")
            
            # Recomendações
            f.write(f"💡 RECOMENDAÇÕES:\n")
            if np.mean(cpu_values) > 80:
                f.write(f"   - CPU usage alto - considere otimizar algoritmos\n")
            if memory_values[-1] - memory_values[0] > 500:
                f.write(f"   - Possível vazamento de memória detectado\n")
            if np.mean(gpu_values) < 50 and self.gpu_available:
                f.write(f"   - GPU subutilizada - considere aumentar batch size\n")
            
        print(f"📋 Relatório final salvo em: {report_file}")
    
    def profile_function(self, func, *args, **kwargs):
        """🎯 Profile uma função específica"""
        if not self.monitoring_active:
            self.start_monitoring()
        
        start_time = time.time()
        
        # Line profiling (se disponível)
        if self.line_profiler and LINE_PROFILER_AVAILABLE:
            self.line_profiler.add_function(func)
            self.line_profiler.enable_by_count()
        
        result = func(*args, **kwargs)
        end_time = time.time()
        
        if self.line_profiler and LINE_PROFILER_AVAILABLE:
            self.line_profiler.disable_by_count()
        
        # Salvar resultados
        func_name = func.__name__
        execution_time = end_time - start_time
        
        if self.line_profiler and LINE_PROFILER_AVAILABLE:
            profile_file = f"{self.output_dir}/function_profile_{func_name}_{self.timestamp}.txt"
            with open(profile_file, 'w') as f:
                self.line_profiler.print_stats(stream=f)
            print(f"📊 Profile detalhado salvo em: {profile_file}")
        
        print(f"🎯 Função {func_name} executada em {execution_time:.3f}s")
        
        # Log no arquivo detalhado
        with open(self.detailed_log, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now()}] FUNCTION_PROFILE {func_name}: {execution_time:.3f}s\n")
        
        return result
    
    def get_current_stats(self):
        """📊 Obter estatísticas atuais"""
        if not self.metrics_buffer:
            return {}
        
        latest = self.metrics_buffer[-1]
        return {
            'cpu_percent': latest.cpu_percent,
            'memory_mb': latest.memory_mb,
            'gpu_utilization': latest.gpu_utilization,
            'gpu_memory_mb': latest.gpu_memory_mb,
            'thread_count': latest.thread_count,
            'elapsed_time': latest.timestamp - self.start_time if hasattr(self, 'start_time') else 0
        }

# Instância global do profiler
profiler = RealTimeProfiler()

def start_profiling():
    """🚀 Iniciar profiling global"""
    profiler.start_monitoring()

def stop_profiling():
    """🛑 Parar profiling global"""
    profiler.stop_monitoring()

def profile_training_step(func):
    """🎯 Decorator para profile de training steps"""
    def wrapper(*args, **kwargs):
        return profiler.profile_function(func, *args, **kwargs)
    return wrapper

def get_profiling_stats():
    """📊 Obter estatísticas atuais"""
    return profiler.get_current_stats()

# Context manager para profiling
class ProfileContext:
    """Context manager para profiling de blocos de código"""
    
    def __init__(self, name="code_block"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"🔍 Iniciando profile: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"⏱️ {self.name} executado em {duration:.3f}s")
        
        # Log no arquivo detalhado
        with open(profiler.detailed_log, 'a') as f:
            f.write(f"[{datetime.now()}] {self.name}: {duration:.3f}s\n")

if __name__ == "__main__":
    print("🔍 DAYPROFILE - Sistema de Profiling em Tempo Real")
    print("=" * 60)
    print("Uso:")
    print("  from dayprofile import start_profiling, stop_profiling, ProfileContext")
    print("  start_profiling()  # Iniciar monitoramento")
    print("  # ... seu código de treinamento ...")
    print("  stop_profiling()   # Parar e gerar relatório")
    print()
    print("Ou use o context manager:")
    print("  with ProfileContext('meu_codigo'):")
    print("      # ... código a ser profileado ...")
    print("=" * 60)
    
    # Teste básico
    print("🧪 Executando teste básico...")
    start_profiling()
    
    # Simular carga de trabalho
    for i in range(10):
        time.sleep(0.1)
        # Simular uso de memória
        data = np.random.random((1000, 1000))
        del data
    
    stop_profiling()
    print("✅ Teste concluído!")