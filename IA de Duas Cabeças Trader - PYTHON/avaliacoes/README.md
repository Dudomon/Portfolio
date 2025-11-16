# 🔥 Sistema de Monitoramento em Tempo Real

Sistema completo de logging JSON e monitoramento tempo real para substituir CSVs pesados e permitir análise instantânea de convergência e gradientes.

## 📁 Estrutura

```
avaliacoes/
├── real_time_logger.py      # Logger JSON streaming principal
├── real_time_monitor.py     # Dashboard interativo tempo real  
├── logger_integration.py    # Integração com sistema existente
├── README.md               # Esta documentação
└── [dados gerados]/
    ├── training_YYYYMMDD_HHMMSS.jsonl     # Dados de treinamento
    ├── gradients_YYYYMMDD_HHMMSS.jsonl    # Informações de gradiente
    ├── convergence_YYYYMMDD_HHMMSS.jsonl  # Métricas de convergência
    ├── rewards_YYYYMMDD_HHMMSS.jsonl      # Dados de reward/episódio
    ├── performance_YYYYMMDD_HHMMSS.jsonl  # Métricas de performance
    └── dashboard_YYYYMMDD_HHMMSS.html     # Dashboard interativo
```

## 🚀 Quick Start

### 1. Usar o Logger Diretamente

```python
from avaliacoes.real_time_logger import create_real_time_logger

# Criar logger
with create_real_time_logger() as logger:
    # Log dados de treinamento
    logger.log_training_step(step=100, loss=0.5, learning_rate=2e-4)
    
    # Log gradientes
    logger.log_gradient_info(step=100, grad_norm=1.2, grad_zeros_ratio=0.1)
    
    # Log episódios
    logger.log_reward_info(step=100, episode_reward=150.0, episode_length=50)
```

### 2. Monitoramento em Tempo Real

```python
from avaliacoes.real_time_monitor import create_monitor

# Criar monitor
monitor = create_monitor(refresh_interval=1.0)

# Iniciar monitoramento (busca automaticamente última sessão)
monitor.start_monitoring()

# Monitor roda em background, criando dashboard interativo
```

### 3. Integração com Sistema Existente

```python
from avaliacoes.logger_integration import create_integrated_logger

# Criar integração transparente
integration = create_integrated_logger()
session_id = integration.start_session("meu_treino")

# Usar normalmente - substitui CSV automaticamente
integration.log_training_step(loss=0.5, lr=2e-4)
integration.log_gradient_info(model=meu_modelo)  # Extrai gradientes automaticamente
```

## 📊 Vantagens sobre CSV

### ❌ Problemas do CSV
- **Travamento**: Arquivo grande trava leitura tempo real
- **Parsing**: Precisa ler arquivo inteiro para dados recentes  
- **Memória**: CSV gigante consome muita RAM
- **Concorrência**: Conflitos entre escrita/leitura simultânea
- **Formato**: Estrutura rígida, difícil extensibilidade

### ✅ Benefícios do JSON Streaming (JSONL)
- **Stream Real-Time**: Leitura linha por linha sem travamento
- **Buffer Inteligente**: Flush automático otimizado para performance
- **Concorrência**: Escrita/leitura simultânea sem conflitos
- **Flexibilidade**: Estrutura JSON permite dados complexos
- **Análise Automática**: Alertas e detecção de problemas em tempo real
- **Dashboard**: Visualização interativa instantânea

## 📈 Features do Sistema

### 🔥 RealTimeLogger
- **JSON Lines (JSONL)**: Formato otimizado para streaming
- **Buffer Circular**: Memória eficiente com flush automático
- **Multi-Categoria**: Diferentes tipos de dados organizados
- **Thread Safety**: Operação segura em ambiente multi-thread
- **Auto-Flush**: Persistência automática configurável
- **Alertas**: Detecção automática de problemas (gradient explosion, etc.)

### 📊 RealTimeMonitor  
- **Dashboard Plotly**: Gráficos interativos profissionais
- **Matplotlib Fallback**: Suporte sem dependências externas
- **Análise Contínua**: Detecção automática de padrões problemáticos
- **Alertas Visuais**: Notificações em tempo real de issues
- **Múltiplas Métricas**: Loss, gradientes, rewards, performance
- **Export**: Relatórios de análise automáticos

### 🔗 LoggerIntegration
- **Patch Transparente**: Integração sem modificar código existente
- **SB3 Callback**: Suporte nativo para Stable-Baselines3
- **CSV Fallback**: Compatibilidade com sistemas legados
- **Extração Automática**: Coleta automática de gradientes do modelo
- **Bridge**: Converte sistemas antigos para novo formato

## 🎯 Casos de Uso

### 1. Monitoramento de Treinamento RL
```python
# Durante treinamento PPO/SAC/etc
integration.log_training_step(
    step=step,
    loss=loss_value,
    policy_loss=policy_loss,
    value_loss=value_loss,
    entropy_loss=entropy_loss,
    learning_rate=lr,
    clipfrac=clipfrac,
    explained_variance=explained_var
)
```

### 2. Análise de Gradientes
```python
# Automático via modelo
integration.log_gradient_info(model=policy_network)

# Manual para componentes específicos  
integration.log_gradient_info(
    component="actor_head",
    grad_norm=norm_value,
    grad_zeros_ratio=zeros_ratio,
    weight_update_ratio=update_ratio
)
```

### 3. Tracking de Performance
```python
# Métricas de episódio
integration.log_episode_end(
    episode_reward=total_reward,
    episode_length=steps,
    win_rate=win_percentage,
    drawdown=max_drawdown,
    trades_count=num_trades
)
```

### 4. Alertas Automáticos
O sistema detecta automaticamente:
- **Gradient Explosion**: `grad_norm > 10.0`
- **Vanishing Gradients**: `grad_norm < 1e-8`  
- **Muitos Zeros**: `zeros_ratio > 50%`
- **Loss Divergence**: Tendência crescente por 50+ steps
- **Stagnation**: Loss sem mudança significativa
- **Poor Performance**: Rewards consistentemente baixos

## 🔧 Configuração Avançada

### Logger Personalizado
```python
logger = RealTimeLogger(
    base_path="meus_logs",
    buffer_size=2000,          # Buffer maior para alta frequência
    flush_interval=0.5         # Flush mais frequente
)
```

### Monitor Personalizado
```python
monitor = RealTimeMonitor(
    log_path="meus_logs", 
    refresh_interval=1.0,      # Atualização a cada 1s
    history_window=2000        # Manter 2000 pontos na memória
)

# Configurar alertas
monitor.plot_config.update({
    'gradient_threshold_high': 8.0,    # Threshold menor
    'convergence_window': 100,         # Janela maior  
    'alert_retention_minutes': 60      # Manter alertas por 1h
})
```

### Integração SB3
```python
from stable_baselines3 import PPO

# Criar callback
integration = create_integrated_logger()
callback = integration.create_sb3_callback()

# Usar no treinamento
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100000, callback=callback)
```

## 📱 Dashboard Interativo

O dashboard gerado automaticamente inclui:

### 📊 Painel 1: Loss & Learning Rate
- Evolução da loss ao longo do tempo
- Learning rate scheduling
- Trends e médias móveis

### 🔥 Painel 2: Gradient Health  
- Norma dos gradientes
- Percentual de zeros
- Alertas visuais para problemas

### 💰 Painel 3: Reward Trends
- Rewards por episódio  
- Médias móveis
- Performance ao longo do tempo

### ⚙️ Painel 4: Training Metrics
- Entropy loss, value loss, policy loss
- Explained variance
- Outras métricas customizadas

## 🚨 Sistema de Alertas

### Níveis de Alerta
- 🔴 **ERROR**: Problemas críticos (gradient explosion, divergence)
- 🟡 **WARNING**: Problemas moderados (muitos zeros, performance baixa)  
- 🔵 **INFO**: Informações relevantes (stagnation, padrões)

### Persistência  
- Alertas ficam visíveis por 30 minutos (configurável)
- Histórico completo salvo nos logs JSON
- Relatórios de análise incluem sumário de alertas

## 🎛️ API Reference

### RealTimeLogger
```python
logger = RealTimeLogger(base_path, buffer_size, flush_interval)
logger.log_training_step(step, **metrics)
logger.log_gradient_info(step, **gradient_data)  
logger.log_convergence_metrics(step, **convergence_data)
logger.log_reward_info(step, **reward_data)
logger.log_performance_metrics(step, **performance_data)
logger.get_real_time_stats()
logger.close()
```

### RealTimeMonitor
```python  
monitor = RealTimeMonitor(log_path, refresh_interval, history_window)
monitor.start_monitoring(session_id)
monitor.stop_monitoring()
monitor.get_current_status()
monitor.export_analysis_report()
```

### LoggerIntegration
```python
integration = TrainingLoggerIntegration(base_path, enable_csv_fallback, gradient_monitoring)
session_id = integration.start_session(prefix)
integration.log_training_step(**kwargs)
integration.log_gradient_info(model, **manual_data)
integration.log_episode_end(**episode_data)  
integration.create_sb3_callback()
integration.patch_existing_logger(logger_instance, method_name)
integration.end_session()
```

## 🔄 Migração do Sistema Atual

### Passo 1: Teste Paralelo
```python
# Manter CSV atual + adicionar JSON
integration = create_integrated_logger(enable_csv_fallback=True)
```

### Passo 2: Validação
```python
# Comparar dados CSV vs JSON
reader = LogReader()
json_data = reader.read_stream('training')
# Verificar consistência
```

### Passo 3: Substituição Gradual
```python  
# Desabilitar CSV quando confiante
integration = create_integrated_logger(enable_csv_fallback=False)
```

### Passo 4: Cleanup
```python
# Remover CSVs antigos
import shutil
shutil.rmtree("logs_csv_antigos")
```

## 📝 Exemplo Completo

```python
#!/usr/bin/env python3
from avaliacoes.logger_integration import create_integrated_logger
from avaliacoes.real_time_monitor import create_monitor
import time
import numpy as np

def exemplo_treinamento():
    # 1. Criar integração
    integration = create_integrated_logger()
    session_id = integration.start_session("exemplo_ppo")
    
    # 2. Iniciar monitor em thread separada
    monitor = create_monitor()
    monitor.start_monitoring(session_id)
    
    try:
        # 3. Simular treinamento
        for step in range(1000):
            # Log dados de treinamento
            integration.log_training_step(
                loss=np.random.uniform(0.5, 2.0),
                policy_loss=np.random.uniform(0.1, 0.8),
                value_loss=np.random.uniform(0.1, 0.5),
                entropy_loss=np.random.uniform(0.01, 0.1),
                learning_rate=2.5e-4 * (0.999 ** (step // 100)),
                clipfrac=np.random.uniform(0.1, 0.3)
            )
            
            # Log gradientes
            integration.log_gradient_info(
                grad_norm=np.random.lognormal(0, 0.5),
                grad_zeros_ratio=np.random.beta(1, 5)
            )
            
            # Log episódios
            if step % 20 == 0:
                integration.log_episode_end(
                    episode_reward=np.random.normal(100, 50),
                    episode_length=np.random.randint(50, 200),
                    win_rate=np.random.beta(3, 2)
                )
            
            time.sleep(0.01)  # Simular tempo de processamento
            
            # Status periódico
            if step % 100 == 0:
                stats = integration.get_current_stats()
                print(f"Step {step}: {stats}")
    
    finally:
        # 4. Cleanup
        monitor.stop_monitoring()
        report_file = monitor.export_analysis_report()
        integration.end_session()
        
        print(f"Treinamento concluído!")
        print(f"Relatório: {report_file}")
        print(f"Dashboard: avaliacoes/dashboard_{session_id}.html")

if __name__ == "__main__":
    exemplo_treinamento()
```

## 🎉 Conclusão

Este sistema resolve completamente o problema de monitoramento em tempo real, substituindo CSVs pesados por JSON streaming eficiente. Permite:

- ✅ **Zero Lag**: Monitoramento instantâneo sem travamentos
- ✅ **Auto-Análise**: Detecção automática de problemas  
- ✅ **Dashboard**: Visualização profissional em tempo real
- ✅ **Integração**: Compatível com código existente
- ✅ **Alertas**: Notificações automáticas de issues críticos
- ✅ **Performance**: Sistema otimizado para alta frequência
- ✅ **Flexibilidade**: Extensível para novos tipos de dados

**🚀 Agora você pode monitorar convergência e gradientes em tempo real!**