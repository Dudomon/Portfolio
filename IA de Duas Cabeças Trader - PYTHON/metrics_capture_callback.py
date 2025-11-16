#!/usr/bin/env python3
"""
🎯 CALLBACK PARA CAPTURAR MÉTRICAS REAIS DO PPO
Captura explained_variance e outras métricas durante o treinamento
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any

class MetricsCaptureCallback(BaseCallback):
    """
    Callback para capturar métricas reais do PPO durante o treinamento
    """
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.latest_metrics = {}
        self.metrics_history = []
        
    def _on_training_start(self) -> None:
        """Chamado no início do treinamento"""
        if self.verbose:
            print("🎯 MetricsCaptureCallback ativado!")
            
    def _on_rollout_end(self) -> None:
        """Chamado no final de cada rollout - aqui temos as métricas reais"""
        try:
            # CORREÇÃO: Capturar métricas do PPO diretamente
            logger_metrics = {}
            
            # Método 1: Logger do modelo (SB3)
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logger_metrics.update(self.model.logger.name_to_value.copy())
            
            # Método 2: Acessar métricas internas do PPO
            if hasattr(self.model, '_last_train_info'):
                info = self.model._last_train_info
                if info and isinstance(info, dict):
                    logger_metrics.update(info)
            
            # Método 3: Buffer de rollouts
            if hasattr(self.model, 'rollout_buffer') and hasattr(self.model.rollout_buffer, 'returns'):
                buffer = self.model.rollout_buffer
                if hasattr(buffer, 'returns') and len(buffer.returns) > 0:
                    logger_metrics['buffer_returns_mean'] = np.mean(buffer.returns)
                    logger_metrics['buffer_returns_std'] = np.std(buffer.returns)
                
                # Processar e limpar nomes das métricas
                processed_metrics = {}
                for key, value in logger_metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        clean_key = key.replace('/', '_').replace('train_', '')
                        processed_metrics[clean_key] = float(value)
                        
                        # Mapear nomes específicos do PPO
                        if 'explained_var' in key.lower():
                            processed_metrics['explained_variance'] = float(value)
                        elif 'policy_loss' in key.lower() or 'policy_gradient_loss' in key.lower():
                            processed_metrics['policy_loss'] = float(value)
                        elif 'value_loss' in key.lower():
                            processed_metrics['value_loss'] = float(value)
                        elif 'entropy_loss' in key.lower():
                            processed_metrics['entropy_loss'] = float(value)
                        elif 'clip_fraction' in key.lower():
                            processed_metrics['clip_fraction'] = float(value)
                        elif 'learning_rate' in key.lower():
                            processed_metrics['learning_rate'] = float(value)
                
                # Salvar métricas mais recentes
                self.latest_metrics = processed_metrics
                
                # Adicionar timestamp e step
                metrics_with_meta = processed_metrics.copy()
                metrics_with_meta['timestep'] = self.num_timesteps
                metrics_with_meta['n_calls'] = self.n_calls
                
                # Manter histórico limitado (últimos 100)
                self.metrics_history.append(metrics_with_meta)
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                    
                # Debug ocasional + FEED JSONL LOGGER
                if self.num_timesteps % 10000 == 0 and self.verbose:
                    exp_var = processed_metrics.get('explained_variance', 0)
                    policy_loss = processed_metrics.get('policy_loss', 0)
                    value_loss = processed_metrics.get('value_loss', 0)
                    clip_frac = processed_metrics.get('clip_fraction', 0)
                    
                    print(f"🎯 [METRICS] Step {self.num_timesteps}: "
                          f"ExpVar={exp_var:.3f}, PolicyLoss={policy_loss:.6f}, "
                          f"ValueLoss={value_loss:.6f}, ClipFrac={clip_frac:.3f}")
                
                # 🔥 FEED JSONL LOGGER - LOGS COMPLETOS
                try:
                    from avaliacoes.real_time_logger import get_logger
                    logger = get_logger()
                    if logger:
                        # 1. Métricas básicas de treinamento
                        logger.log_training_step(self.num_timesteps, processed_metrics)
                        
                        # 2. Métricas detalhadas do critic
                        critic_metrics = {
                            'critic_loss': processed_metrics.get('value_loss', 0),
                            'critic_explained_var': processed_metrics.get('explained_variance', 0),
                            'critic_lr': processed_metrics.get('learning_rate', 0),
                            'returns_mean': processed_metrics.get('buffer_returns_mean', 0),
                            'returns_std': processed_metrics.get('buffer_returns_std', 0)
                        }
                        logger.log_performance_metric(self.num_timesteps, 'critic', critic_metrics)
                        
                        # 3. Métricas detalhadas do actor
                        actor_metrics = {
                            'actor_loss': processed_metrics.get('policy_loss', 0),
                            'entropy_loss': processed_metrics.get('entropy_loss', 0),
                            'clip_fraction': processed_metrics.get('clip_fraction', 0),
                            'kl_divergence': processed_metrics.get('kl_divergence', 0),
                            'actor_lr': processed_metrics.get('learning_rate', 0)
                        }
                        logger.log_performance_metric(self.num_timesteps, 'actor', actor_metrics)
                        
                        # 4. Métricas de reward (quando disponíveis)
                        if hasattr(self.model, 'env') and hasattr(self.model.env, 'get_attr'):
                            try:
                                episode_rewards = self.model.env.get_attr('episode_returns')
                                if episode_rewards and any(episode_rewards):
                                    avg_reward = sum([r for r in episode_rewards if r is not None]) / len([r for r in episode_rewards if r is not None])
                                    reward_metrics = {
                                        'episode_reward_mean': avg_reward,
                                        'episode_count': len([r for r in episode_rewards if r is not None])
                                    }
                                    logger.log_performance_metric(self.num_timesteps, 'rewards', reward_metrics)
                            except:
                                pass
                        
                except Exception as logger_err:
                    if self.verbose and self.num_timesteps % 50000 == 0:
                        print(f"⚠️ JSONL Logger não conectado: {logger_err}")
                    
        except Exception as e:
            if self.verbose:
                print(f"⚠️ [METRICS] Erro ao capturar métricas: {e}")
    
    def _on_step(self) -> bool:
        """Chamado a cada step"""
        return True
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Retorna as métricas mais recentes capturadas"""
        return self.latest_metrics.copy()
    
    def get_metric(self, metric_name: str, default=0):
        """Retorna uma métrica específica"""
        return self.latest_metrics.get(metric_name, default)
    
    def has_metrics(self) -> bool:
        """Verifica se há métricas disponíveis"""
        return len(self.latest_metrics) > 0

# Factory function para fácil uso
def create_metrics_capture_callback(verbose=1):
    """Cria uma instância do callback"""
    return MetricsCaptureCallback(verbose=verbose)