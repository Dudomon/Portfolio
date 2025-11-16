# -*- coding: utf-8 -*-
"""
Enhanced Running Normalizer - Sistema unico de normalizacao
Versão compatível com treinamento diferenciado PPO
"""

import numpy as np
import pickle
import os
from typing import Optional, Dict, Any
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
import gym


class EnhancedRunningNormalizer:
    """
    Enhanced Running Normalizer - Sistema unico de normalizacao
    Compatível com VecNormalize mas com funcionalidades avançadas
    """
    
    def __init__(self, 
                 obs_size: int = 1320,  # 🔧 CORRIGIDO: Compatible com PPOV1 (66 features × 20 window)
                 training: bool = True,
                 norm_obs: bool = True,
                 norm_reward: bool = True,
                 clip_obs: float = 2.0,
                 clip_reward: float = 5.0,
                 gamma: float = 0.99,
                 epsilon: float = 1e-6,
                 momentum: float = 0.999,
                 warmup_steps: int = 2000,
                 stability_check: bool = True):
        
        self.obs_size = obs_size
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.stability_check = stability_check
        
        # Estatísticas de observação
        self.obs_rms = RunningMeanStd(shape=(obs_size,), epsilon=epsilon)
        
        # Estatísticas de recompensa
        self.ret_rms = RunningMeanStd(shape=(), epsilon=epsilon)
        
        # Contadores
        self.step_count = 0
        self.warmup_complete = False
        
        # Histórico para estabilidade
        self.obs_history = []
        self.reward_history = []
        
        print(f"Enhanced Running Normalizer criado: obs_size={obs_size}, training={training}")
    
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normaliza observações"""
        if not self.norm_obs or not self.warmup_complete:
            return obs
        
        # Atualizar estatísticas se em treinamento
        if self.training:
            self.obs_rms.update(obs)
        
        # Normalizar
        normalized_obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        
        # Clipping
        if self.clip_obs > 0:
            normalized_obs = np.clip(normalized_obs, -self.clip_obs, self.clip_obs)
        
        return normalized_obs
    
    def normalize_reward(self, reward: float) -> float:
        """Normaliza recompensas"""
        if not self.norm_reward or not self.warmup_complete:
            return reward
        
        # Atualizar estatísticas se em treinamento
        if self.training:
            self.ret_rms.update(np.array([reward]))
        
        # Normalizar
        normalized_reward = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        
        # Clipping
        if self.clip_reward > 0:
            normalized_reward = np.clip(normalized_reward, -self.clip_reward, self.clip_reward)
        
        return normalized_reward
    
    def step(self, obs: np.ndarray, reward: float) -> tuple:
        """Processa um step de normalização"""
        self.step_count += 1
        
        # Verificar se warmup está completo
        if self.step_count >= self.warmup_steps and not self.warmup_complete:
            self.warmup_complete = True
            print(f"Enhanced Normalizer warmup completo apos {self.warmup_steps} steps")
        
        # Normalizar
        norm_obs = self.normalize_obs(obs)
        norm_reward = self.normalize_reward(reward)
        
        # Histórico para estabilidade
        if self.stability_check:
            self.obs_history.append(norm_obs.copy())
            self.reward_history.append(norm_reward)
            
            # Manter apenas últimos 1000 valores
            if len(self.obs_history) > 1000:
                self.obs_history.pop(0)
                self.reward_history.pop(0)
        
        return norm_obs, norm_reward
    
    def reset(self) -> None:
        """Reseta o normalizer"""
        self.obs_rms.reset()
        self.ret_rms.reset()
        self.step_count = 0
        self.warmup_complete = False
        self.obs_history.clear()
        self.reward_history.clear()
        print("Enhanced Normalizer resetado")
    
    def save(self, filepath: str) -> bool:
        """Salva o normalizer"""
        try:
            data = {
                'obs_rms_mean': self.obs_rms.mean,
                'obs_rms_var': self.obs_rms.var,
                'obs_rms_count': self.obs_rms.count,
                'ret_rms_mean': self.ret_rms.mean,
                'ret_rms_var': self.ret_rms.var,
                'ret_rms_count': self.ret_rms.count,
                'step_count': self.step_count,
                'warmup_complete': self.warmup_complete,
                'config': {
                    'obs_size': self.obs_size,
                    'norm_obs': self.norm_obs,
                    'norm_reward': self.norm_reward,
                    'clip_obs': self.clip_obs,
                    'clip_reward': self.clip_reward,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon,
                    'momentum': self.momentum,
                    'warmup_steps': self.warmup_steps
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Enhanced Normalizer salvo: {filepath}")
            return True
            
        except Exception as e:
            print(f"Erro ao salvar Enhanced Normalizer: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str, env=None) -> 'EnhancedRunningNormalizer':
        """Carrega o normalizer"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Verificar se é formato novo (com config) ou antigo
            if isinstance(data, dict) and 'config' in data:
                # Formato novo - usar configurações salvas
                config = data['config']
                normalizer = cls(
                    obs_size=config['obs_size'],
                    training=False,  # Modo produção
                    norm_obs=config.get('norm_obs', True),
                    norm_reward=config.get('norm_reward', True),
                    clip_obs=config.get('clip_obs', 2.0),
                    clip_reward=config.get('clip_reward', 5.0),
                    gamma=config.get('gamma', 0.99),
                    epsilon=config.get('epsilon', 1e-6),
                    momentum=config.get('momentum', 0.999),
                    warmup_steps=config.get('warmup_steps', 2000)
                )
                
                # Restaurar estatísticas se disponíveis
                if 'obs_rms_mean' in data:
                    normalizer.obs_rms.mean = data['obs_rms_mean']
                    normalizer.obs_rms.var = data['obs_rms_var']
                    normalizer.obs_rms.count = data['obs_rms_count']
                    normalizer.ret_rms.mean = data['ret_rms_mean']
                    normalizer.ret_rms.var = data['ret_rms_var']
                    normalizer.ret_rms.count = data['ret_rms_count']
                    normalizer.step_count = data.get('step_count', 0)
                    normalizer.warmup_complete = data.get('warmup_complete', True)
                    print(f"Estatisticas Enhanced Normalizer restauradas: count={normalizer.obs_rms.count}")
                
            else:
                # FORMATO INVÁLIDO - APENAS V5/V6 ACEITOS
                raise Exception(f"Enhanced Normalizer inválido: {filepath} - Apenas formatos V5/V6 aceitos")
            
            print(f"Enhanced Normalizer carregado: {filepath}")
            return normalizer
            
        except Exception as e:
            print(f"Erro ao carregar Enhanced Normalizer: {e}")
            raise Exception(f"Enhanced Normalizer inválido ou corrompido: {filepath} - Sistema aceita apenas V5 (1320 obs) ou V6 (1480 obs)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do normalizer"""
        return {
            'obs_mean': self.obs_rms.mean.mean() if hasattr(self.obs_rms.mean, 'mean') else float(self.obs_rms.mean),
            'obs_std': np.sqrt(self.obs_rms.var).mean() if hasattr(self.obs_rms.var, 'mean') else float(np.sqrt(self.obs_rms.var)),
            'reward_mean': float(self.ret_rms.mean),
            'reward_std': float(np.sqrt(self.ret_rms.var)),
            'step_count': self.step_count,
            'warmup_complete': self.warmup_complete
        }
    
    def check_health(self, obs: np.ndarray) -> tuple:
        """Verifica saúde do normalizer"""
        try:
            obs_flat = obs.flatten()
            
            # Estatísticas das observações
            obs_mean = np.mean(obs_flat)
            obs_std = np.std(obs_flat)
            obs_min = np.min(obs_flat)
            obs_max = np.max(obs_flat)
            
            # Detectar problemas de normalização
            near_zero = np.sum(np.abs(obs_flat) < 0.01) / len(obs_flat)
            extreme_values = np.sum(np.abs(obs_flat) > 5.0) / len(obs_flat)
            nan_count = np.sum(np.isnan(obs_flat))
            
            # Determinar se está saudável
            is_healthy = near_zero < 0.7 and extreme_values < 0.15 and nan_count == 0
            
            health_info = {
                'is_healthy': is_healthy,
                'near_zero_ratio': near_zero,
                'extreme_ratio': extreme_values,
                'nan_count': nan_count,
                'obs_mean': obs_mean,
                'obs_std': obs_std,
                'obs_range': [obs_min, obs_max]
            }
            
            return health_info, obs_flat
            
        except Exception as e:
            print(f"Erro ao verificar saude do Enhanced Normalizer: {e}")
            return {'is_healthy': False, 'error': str(e)}, obs
    
    def adapt_to_production(self, production_obs: np.ndarray, adaptation_strength: float = 0.1) -> np.ndarray:
        """🔧 ADAPTAÇÃO PARA PRODUÇÃO: Ajusta normalizer para dados MT5 reais"""
        try:
            if not self.training:
                # Temporariamente ativar treinamento para adaptação
                original_training = False
                self.training = True
            else:
                original_training = True
            
            # Calcular estatísticas dos dados de produção
            prod_mean = np.mean(production_obs)
            prod_std = np.std(production_obs)
            
            # Ajustar gradualmente as estatísticas (momentum adaptativo)
            if self.obs_rms.count > 0:
                # Misturar estatísticas existentes com dados de produção
                current_mean = self.obs_rms.mean
                current_var = self.obs_rms.var
                
                # Adaptação suave (weighted average)
                weight = adaptation_strength
                adapted_mean = (1 - weight) * current_mean + weight * prod_mean
                adapted_var = (1 - weight) * current_var + weight * (prod_std ** 2)
                
                # Atualizar estatísticas
                self.obs_rms.mean = adapted_mean
                self.obs_rms.var = adapted_var
                
                # Adaptação silenciosa para produção
            
            # Normalizar com estatísticas adaptadas
            normalized_obs = self.normalize_obs(production_obs)
            
            # Restaurar estado original
            self.training = original_training
            
            return normalized_obs
            
        except Exception as e:
            print(f"Erro na adaptacao para producao: {e}")
            return self.normalize_obs(production_obs)
    
    def detect_and_fix_anomalies(self, obs: np.ndarray, fix_threshold: float = 0.2) -> tuple:
        """🔧 DETECÇÃO E CORREÇÃO DE ANOMALIAS: Corrige observações anômalas"""
        try:
            obs_flat = obs.flatten()
            
            # 🔧 CORREÇÃO: Determinar se dados são brutos ou normalizados
            obs_mean = np.mean(obs_flat)
            obs_std = np.std(obs_flat)
            
            # Se mean > 1 ou std > 1, provavelmente são dados brutos
            if abs(obs_mean) > 1.0 or obs_std > 1.0:
                # Thresholds para dados brutos
                tiny_mask = np.abs(obs_flat) < 1e-8  # Zeros extremos
                huge_mask = np.abs(obs_flat) > 100.0  # Valores muito grandes para dados brutos
                anomaly_mask = tiny_mask | huge_mask
            else:
                # Thresholds para dados normalizados
                tiny_mask = np.abs(obs_flat) < 1e-8  # Zeros extremos
                huge_mask = np.abs(obs_flat) > 5.0   # Valores além de 5 sigmas
                anomaly_mask = tiny_mask | huge_mask
            
            anomaly_count = np.sum(anomaly_mask)
            anomaly_ratio = anomaly_count / len(obs_flat)
            
            # Se há muitas anomalias, corrigir
            if anomaly_ratio > fix_threshold:
                
                # Corrigir valores muito pequenos
                obs_flat[tiny_mask] = np.random.normal(0, 0.01, np.sum(tiny_mask))
                
                # Corrigir valores muito grandes
                obs_flat[huge_mask] = np.random.normal(0, 2.0, np.sum(huge_mask))
                
                # Reshape de volta ao formato original
                corrected_obs = obs_flat.reshape(obs.shape)
                
                return corrected_obs, anomaly_count
            
            return obs, anomaly_count
            
        except Exception as e:
            print(f"Erro na correcao de anomalias: {e}")
            return obs, 0


class RunningMeanStd:
    """Implementação de Running Mean/Std compatível com VecNormalize"""
    
    def __init__(self, shape, epsilon=1e-6):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.epsilon = epsilon
    
    def update(self, x):
        """Atualiza estatísticas com novos dados"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 1 else 1
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def reset(self):
        """Reseta estatísticas"""
        self.mean.fill(0)
        self.var.fill(1)
        self.count = 0


def create_enhanced_normalizer(env,
                              obs_size=None,
                              training=True,
                              norm_obs=True,
                              norm_reward=True,
                              clip_obs=2.0,
                              clip_reward=5.0,
                              gamma=0.99,
                              epsilon=1e-6,
                              momentum=0.999,
                              warmup_steps=2000,
                              stability_check=True):
    """
    Cria um Enhanced Running Normalizer
    
    Args:
        env: Ambiente a ser normalizado
        obs_size: Tamanho das observações
        training: Se está em modo treinamento
        norm_obs: Se normalizar observações
        norm_reward: Se normalizar recompensas
        clip_obs: Clipping para observações
        clip_reward: Clipping para recompensas
        gamma: Fator de desconto
        epsilon: Epsilon para estabilidade numérica
        momentum: Momentum para atualizações
        warmup_steps: Steps de aquecimento
        stability_check: Se verificar estabilidade
    
    Returns:
        EnhancedRunningNormalizer configurado
    """
    
    # Se env é um VecEnv, extrair o ambiente base
    if hasattr(env, 'venv'):
        base_env = env.venv.envs[0]
    elif hasattr(env, 'envs'):
        base_env = env.envs[0]
    else:
        base_env = env
    
    # Criar normalizer
    normalizer = EnhancedRunningNormalizer(
        obs_size=obs_size,
        training=training,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        clip_reward=clip_reward,
        gamma=gamma,
        epsilon=epsilon,
        momentum=momentum,
        warmup_steps=warmup_steps,
        stability_check=stability_check
    )
    
    # Wrapper para compatibilidade com VecEnv
    class EnhancedNormalizerWrapper(gym.Wrapper):
        def __init__(self, env, normalizer):
            super().__init__(env)
            self.normalizer = normalizer
        
        def reset(self, **kwargs):
            obs = self.env.reset(**kwargs)
            if isinstance(obs, tuple):
                obs = obs[0]
            return self.normalizer.normalize_obs(obs)
        
        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            norm_obs, norm_reward = self.normalizer.step(obs, reward)
            return norm_obs, norm_reward, done, info
        
        def save(self, filepath):
            return self.normalizer.save(filepath)
        
        @classmethod
        def load(cls, filepath, env):
            normalizer = EnhancedRunningNormalizer.load(filepath, env)
            return cls(env, normalizer)
    
    # Retornar wrapper
    return EnhancedNormalizerWrapper(base_env, normalizer)


# Função de conveniência para criar wrapper
def create_enhanced_normalizer_wrapper(env, obs_size=None, normalizer_file=None):
    """Wrapper para compatibilidade com código existente"""
    if normalizer_file and os.path.exists(normalizer_file):
        print(f"Carregando Enhanced Normalizer existente: {normalizer_file}")
        try:
            return EnhancedRunningNormalizer.load(normalizer_file, env)
        except Exception as e:
            print(f"Erro ao carregar Enhanced Normalizer: {e}")
            print("Criando novo Enhanced Normalizer...")
    
        # 🔥 CORREÇÃO CRÍTICA: Usar obs_size do ambiente se não fornecido
    if obs_size is None:
        obs_size = env.observation_space.shape[0]
        print(f"🔧 Obs_size automático detectado: {obs_size}")
    
    return create_enhanced_normalizer(
        env,
        obs_size=obs_size,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=2.0,
        clip_reward=5.0,
        gamma=0.99,
        epsilon=1e-6,
        momentum=0.999,
        warmup_steps=2000,
        stability_check=True
    )


def save_enhanced_normalizer(enhanced_env, filepath):
    """Salva Enhanced Normalizer"""
    if hasattr(enhanced_env, 'save'):
        return enhanced_env.save(filepath)
    elif hasattr(enhanced_env, 'normalizer'):
        return enhanced_env.normalizer.save(filepath)
    else:
        print("Enhanced Normalizer nao encontrado para salvar")
        return False


def monitor_enhanced_normalizer_health(enhanced_env, obs):
    """Monitora saúde do Enhanced Normalizer"""
    try:
        if hasattr(enhanced_env, 'normalizer'):
            normalizer = enhanced_env.normalizer
        else:
            return True  # Não há normalizer para monitorar
        
        # Verificar se observações estão sendo normalizadas corretamente
        obs_flat = obs.flatten()
        
        # Estatísticas das observações
        obs_mean = np.mean(obs_flat)
        obs_std = np.std(obs_flat)
        obs_min = np.min(obs_flat)
        obs_max = np.max(obs_flat)
        
        # Detectar problemas de normalização
        near_zero = np.sum(np.abs(obs_flat) < 0.01) / len(obs_flat)
        extreme_values = np.sum(np.abs(obs_flat) > 5.0) / len(obs_flat)
        
        # Alertar se há problemas
        if near_zero > 0.7:
            print(f"ALERTA Enhanced Normalizer: {near_zero*100:.1f}% das observacoes proximas de zero!")
            print(f"   Mean: {obs_mean:.4f}, Std: {obs_std:.4f}, Range: [{obs_min:.4f}, {obs_max:.4f}]")
            return False
        
        if extreme_values > 0.15:
            print(f"ALERTA Enhanced Normalizer: {extreme_values*100:.1f}% valores extremos!")
            return False
        
        return True
    except Exception as e:
        print(f"Erro ao monitorar Enhanced Normalizer: {e}")
        return False


# ALIAS PARA COMPATIBILIDADE COM PICKLE ANTIGO
# O arquivo pickle contém 'EnhancedVecNormalize' mas o arquivo local tem 'EnhancedRunningNormalizer'
EnhancedVecNormalize = EnhancedRunningNormalizer

print("Alias EnhancedVecNormalize -> EnhancedRunningNormalizer criado para compatibilidade") 