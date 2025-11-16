# 🔍 CONFIGURAÇÃO DO PROFILING
# Ajuste estas configurações conforme necessário

PROFILING_CONFIG = {
    # Intervalo de coleta de métricas (segundos)
    "profile_interval": 0.1,  # 100ms - alta frequência
    
    # Intervalo de análise detalhada (segundos)
    "detailed_interval": 1.0,  # 1s - análises complexas
    
    # Tamanho do buffer de métricas
    "buffer_size": 10000,  # ~16 minutos de dados
    
    # Alertas
    "cpu_alert_threshold": 90,  # %
    "memory_alert_threshold": 85,  # %
    "gpu_memory_alert_mb": 8000,  # MB
    
    # Arquivos de output
    "output_dir": "profiling_results",
    "save_detailed_logs": True,
    "save_csv_metrics": True,
    
    # Funções para profile detalhado
    "profile_functions": [
        "model.learn",
        "env.step", 
        "model.predict",
        "calculate_reward",
        "process_observation",
        "_process_v7_action",
        "_check_entry_filters",
        "_apply_v7_intuition_filters"
    ],
    
    # GPU monitoring
    "enable_gpu_monitoring": True,
    
    # Memory leak detection
    "memory_leak_threshold_mb": 100,  # MB growth
    "memory_leak_window": 20,  # samples
}
