import cProfile
import pstats
import io
import time
import threading
import json
from datetime import datetime
from collections import defaultdict, deque

class RealTimeProfiler:
    """
    🔍 PROFILER EM TEMPO REAL
    
    Monitora performance em tempo real e identifica gargalos específicos
    """
    
    def __init__(self, output_file="performance_profile.json"):
        self.output_file = output_file
        self.profiler = cProfile.Profile()
        self.is_profiling = False
        self.start_time = None
        self.function_times = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.recent_stats = deque(maxlen=100)  # Últimas 100 medições
        
        # Métricas específicas para daytrader
        self.observation_times = deque(maxlen=50)
        self.step_times = deque(maxlen=50)
        self.feature_generation_times = deque(maxlen=50)
        
    def start_profiling(self):
        """Iniciar profiling"""
        print(f"🔍 [PROFILER] Iniciando profiling em tempo real...")
        self.profiler.enable()
        self.is_profiling = True
        self.start_time = time.time()
        
        # Thread para análise contínua
        self.analysis_thread = threading.Thread(target=self._continuous_analysis, daemon=True)
        self.analysis_thread.start()
        
    def stop_profiling(self):
        """Parar profiling"""
        if self.is_profiling:
            self.profiler.disable()
            self.is_profiling = False
            print(f"🔍 [PROFILER] Profiling parado")
            
    def profile_function(self, func_name):
        """Decorator para profilear funções específicas"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                self.function_times[func_name].append(execution_time)
                self.call_counts[func_name] += 1
                
                # Métricas específicas
                if func_name.endswith('_get_observation'):
                    self.observation_times.append(execution_time)
                elif func_name.endswith('step'):
                    self.step_times.append(execution_time)
                elif 'feature' in func_name.lower():
                    self.feature_generation_times.append(execution_time)
                
                # Log se for muito lento
                if execution_time > 0.01:  # > 10ms
                    print(f"⚠️ [SLOW] {func_name}: {execution_time*1000:.1f}ms")
                
                return result
            return wrapper
        return decorator
    
    def _continuous_analysis(self):
        """Análise contínua das métricas"""
        analysis_counter = 0
        while self.is_profiling:
            time.sleep(5)  # Análise a cada 5 segundos
            analysis_counter += 1
            # Mostrar apenas a cada 2000 steps (aproximadamente a cada 40 análises)
            if analysis_counter % 8 == 0:  # ~40s = 8 análises de 5s
                self._analyze_current_performance()
    
    def _analyze_current_performance(self):
        """Analisar performance atual"""
        if not self.function_times:
            return
            
        print(f"\n🔍 [PROFILER] ===== ANÁLISE EM TEMPO REAL =====")
        
        # Top funções mais lentas
        avg_times = {}
        for func_name, times in self.function_times.items():
            if times:
                avg_times[func_name] = {
                    'avg_time': sum(times) / len(times),
                    'total_time': sum(times),
                    'call_count': self.call_counts[func_name],
                    'max_time': max(times)
                }
        
        # Ordenar por tempo total gasto
        sorted_funcs = sorted(avg_times.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        print(f"🔥 TOP 5 GARGALOS:")
        for i, (func_name, stats) in enumerate(sorted_funcs[:5]):
            print(f"  {i+1}. {func_name}")
            print(f"     Tempo Total: {stats['total_time']*1000:.1f}ms")
            print(f"     Tempo Médio: {stats['avg_time']*1000:.1f}ms")
            print(f"     Chamadas: {stats['call_count']}")
            print(f"     Tempo Máx: {stats['max_time']*1000:.1f}ms")
        
        # Métricas específicas do daytrader
        if self.observation_times:
            avg_obs = sum(self.observation_times) / len(self.observation_times)
            print(f"\n📊 MÉTRICAS DAYTRADER:")
            print(f"   Observation Time: {avg_obs*1000:.1f}ms (última: {self.observation_times[-1]*1000:.1f}ms)")
        
        if self.step_times:
            avg_step = sum(self.step_times) / len(self.step_times)
            print(f"   Step Time: {avg_step*1000:.1f}ms (última: {self.step_times[-1]*1000:.1f}ms)")
            
        if self.feature_generation_times:
            avg_feature = sum(self.feature_generation_times) / len(self.feature_generation_times)
            print(f"   Feature Generation: {avg_feature*1000:.1f}ms (última: {self.feature_generation_times[-1]*1000:.1f}ms)")
        
        # Performance estimada (it/s)
        if self.step_times:
            current_step_time = self.step_times[-1] if self.step_times else 0.02
            estimated_its = 1.0 / current_step_time if current_step_time > 0 else 0
            print(f"   🚀 Performance Estimada: {estimated_its:.1f} it/s")
        
        print(f"================================================\n")
        
        # Salvar estatísticas
        self._save_stats(avg_times)
    
    def _save_stats(self, avg_times):
        """Salvar estatísticas em arquivo JSON"""
        stats_data = {
            'timestamp': datetime.now().isoformat(),
            'function_stats': avg_times,
            'recent_observations': list(self.observation_times)[-10:] if self.observation_times else [],
            'recent_steps': list(self.step_times)[-10:] if self.step_times else [],
            'estimated_performance': 1.0 / self.step_times[-1] if self.step_times and self.step_times[-1] > 0 else 0
        }
        
        try:
            with open(self.output_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Erro ao salvar stats: {e}")
    
    def get_detailed_stats(self):
        """Obter estatísticas detalhadas usando cProfile"""
        if not self.is_profiling:
            return None
            
        # Criar string buffer para capturar output
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 funções
        
        profile_output = s.getvalue()
        
        print(f"\n🔍 [PROFILER] ESTATÍSTICAS DETALHADAS:")
        print(profile_output)
        
        return profile_output

# Singleton profiler
profiler = RealTimeProfiler()