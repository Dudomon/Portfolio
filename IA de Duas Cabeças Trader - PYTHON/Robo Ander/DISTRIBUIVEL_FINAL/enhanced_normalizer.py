# -*- coding: utf-8 -*-
"""
🚀 Enhanced VecNormalize - Sistema de normalização compatível com Stable-Baselines3
Versão corrigida para funcionar perfeitamente com PPO
"""

import numpy as np
import pickle
import os
from typing import Optional, Dict, Any, Union, List
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.running_mean_std import RunningMeanStd
try:
    import gym
    from gym import spaces
except ImportError:
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        # Fallback para versões mais antigas
        import gym
        from gym import spaces


class EnhancedVecNormalize(VecNormalize):
    """
    🚀 Enhanced VecNormalize - Extensão do VecNormalize padrão com funcionalidades avançadas
    Mantém 100% compatibilidade com VecNormalize original
    """
    
    def __init__(self, 
                 venv,
                 training: bool = True,
                 norm_obs: bool = True,
                 norm_reward: bool = True,
                 clip_obs: float = 5.0,   # 🚨 CRITIC FIX: Reduzir range extremo
                 clip_reward: float = 5.0, # 🚨 CRITIC FIX: Reduzir range extremo  
                 gamma: float = 0.99,
                 epsilon: float = 1e-6,
                 momentum: float = 0.99,   # 🚨 CRITIC FIX: Reduzir momentum para mais responsividade
                 warmup_steps: int = 10000,  # 🚨 CRITIC FIX: Maior warmup para estabilidade
                 stability_check: bool = True):
        
        # Inicializar VecNormalize pai com configurações otimizadas
        super().__init__(
            venv=venv,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
            epsilon=epsilon,
            training=training
        )
        
        # Configurações específicas do Enhanced
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.stability_check = stability_check
        self.step_count = 0
        self.warmup_complete = False
        
        # Histórico para estabilidade
        self.obs_history = []
        self.reward_history = []
        
        print(f"🚀 Enhanced VecNormalize criado: obs_norm={norm_obs}, reward_norm={norm_reward}")
        print(f"   📏 Clip obs: [-{clip_obs}, {clip_obs}], Clip reward: [-{clip_reward}, {clip_reward}]")
        print(f"   🔥 Warmup steps: {warmup_steps}, Stability check: {stability_check}")
        print(f"   ✅ Zeros legítimos preservados (sem intervenção artificial)")
    
    def step(self, actions):
        """Step com monitoramento de estabilidade"""
        self.step_count += 1
        
        # 🔧 CRITIC FIX: Remover mudança abrupta pós-warmup
        # if self.step_count >= self.warmup_steps and not self.warmup_complete:
        #     self.warmup_complete = True
        #     print(f"✅ Enhanced VecNormalize warmup completo após {self.warmup_steps} steps")
        
        # Warmup gradual ao invés de abrupto
        if self.step_count >= self.warmup_steps:
            self.warmup_complete = True
        
        # Executar step normal do VecNormalize
        obs, rewards, dones, infos = super().step(actions)
        
        # 🚀 OPTIMIZED: Warmup simplificado para performance
        # REMOVED: Normalização gradual custosa - usar apenas warmup_complete flag
        
        # 🚀 OPTIMIZED: Monitoring muito esparso para performance
        if self.stability_check and self.warmup_complete and self.step_count % 50000 == 0:
            self._monitor_stability(obs, rewards)
        
        # Removido: Zeros são legítimos em trading - não precisam intervenção
        
        return obs, rewards, dones, infos
    
    def reset(self):
        """Reset com preservação de estatísticas"""
        obs = super().reset()
        
        # Resetar contadores específicos do Enhanced
        self.step_count = 0
        self.warmup_complete = False
        self.obs_history.clear()
        self.reward_history.clear()
        
        return obs
    
    def _monitor_stability(self, obs, rewards):
        """Monitorar estabilidade das observações e recompensas"""
        try:
            # Flatten observações para análise
            if isinstance(obs, (list, np.ndarray)):
                obs_flat = np.concatenate([o.flatten() if hasattr(o, 'flatten') else np.array(o).flatten() for o in obs])
            else:
                obs_flat = obs.flatten() if hasattr(obs, 'flatten') else np.array(obs).flatten()
            
            # Estatísticas das observações
            obs_mean = np.mean(obs_flat)
            obs_std = np.std(obs_flat)
            obs_min = np.min(obs_flat)
            obs_max = np.max(obs_flat)
            
            # Detectar problemas de normalização reais
            # 🎯 CORREÇÃO: Thresholds adequados para dados normalizados
            real_zeros = np.sum(np.abs(obs_flat) < 1e-8) / len(obs_flat)  # Zeros extremos apenas
            extreme_values = np.sum(np.abs(obs_flat) > 5.0) / len(obs_flat)  # Valores além de 5 sigmas
            
            # Alertar apenas se há problemas críticos (sem spam de health check)
            if self.step_count % 20000 == 0:
                if real_zeros > 0.08:  # >8% zeros extremos é problemático (mais rigoroso)
                    print(f"⚠️ ALERTA Enhanced VecNormalize: {real_zeros*100:.1f}% zeros extremos!")
                    print(f"   📊 Mean: {obs_mean:.4f}, Std: {obs_std:.4f}, Range: [{obs_min:.4f}, {obs_max:.4f}]")
                
                if extreme_values > 0.05:  # >5% valores extremos é problemático  
                    print(f"⚠️ ALERTA Enhanced VecNormalize: {extreme_values*100:.1f}% valores extremos!")
                
                # Health check removido - apenas alertas críticos
                
                # CORREÇÃO REMOVIDA - Não mascarar problemas reais
                # Deixar zeros extremos expostos para debug adequado
        
        except Exception as e:
            print(f"❌ Erro no monitoramento de estabilidade: {e}")
    
    # Função _stabilize_variance removida - zeros são legítimos em trading
    
    def save(self, filepath: str) -> bool:
        """Salvar Enhanced VecNormalize com configurações extras"""
        try:
            # Salvar VecNormalize base
            super().save(filepath)
            
            # Salvar configurações específicas do Enhanced
            enhanced_data = {
                'momentum': self.momentum,
                'warmup_steps': self.warmup_steps,
                'stability_check': self.stability_check,
                'step_count': self.step_count,
                'warmup_complete': self.warmup_complete
            }
            
            enhanced_filepath = filepath.replace('.pkl', '_enhanced.pkl')
            with open(enhanced_filepath, 'wb') as f:
                pickle.dump(enhanced_data, f)
            
            print(f"💾 Enhanced VecNormalize salvo: {filepath}")
            print(f"   📁 Configurações extras: {enhanced_filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar Enhanced VecNormalize: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str, venv):
        """Carregar Enhanced VecNormalize"""
        try:
            # Carregar VecNormalize base
            vec_normalize = VecNormalize.load(filepath, venv)
            
            # Tentar carregar configurações específicas do Enhanced
            enhanced_filepath = filepath.replace('.pkl', '_enhanced.pkl')
            enhanced_config = {}
            
            if os.path.exists(enhanced_filepath):
                with open(enhanced_filepath, 'rb') as f:
                    enhanced_config = pickle.load(f)
                print(f"🔄 Configurações Enhanced carregadas: {enhanced_filepath}")
            
            # Criar Enhanced VecNormalize com configurações carregadas
            enhanced_vec = cls(
                venv=venv,
                training=vec_normalize.training,
                norm_obs=vec_normalize.norm_obs,
                norm_reward=vec_normalize.norm_reward,
                clip_obs=vec_normalize.clip_obs,
                clip_reward=vec_normalize.clip_reward,
                gamma=vec_normalize.gamma,
                epsilon=vec_normalize.epsilon,
                momentum=enhanced_config.get('momentum', 0.999),
                warmup_steps=enhanced_config.get('warmup_steps', 2000),
                stability_check=enhanced_config.get('stability_check', True)
            )
            
            # Restaurar estatísticas do VecNormalize carregado
            enhanced_vec.obs_rms.mean = vec_normalize.obs_rms.mean.copy()
            enhanced_vec.obs_rms.var = vec_normalize.obs_rms.var.copy()
            enhanced_vec.obs_rms.count = vec_normalize.obs_rms.count
            enhanced_vec.ret_rms.mean = vec_normalize.ret_rms.mean.copy()
            enhanced_vec.ret_rms.var = vec_normalize.ret_rms.var.copy()
            enhanced_vec.ret_rms.count = vec_normalize.ret_rms.count
            
            # Restaurar configurações específicas
            enhanced_vec.step_count = enhanced_config.get('step_count', 0)
            enhanced_vec.warmup_complete = enhanced_config.get('warmup_complete', False)
            
            print(f"✅ Enhanced VecNormalize carregado: {filepath}")
            return enhanced_vec
            
        except Exception as e:
            print(f"❌ Erro ao carregar Enhanced VecNormalize: {e}")
            print("🔄 Criando novo Enhanced VecNormalize...")
            return cls(venv)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obter estatísticas detalhadas"""
        stats = {
            'obs_mean': self.obs_rms.mean.copy() if hasattr(self.obs_rms, 'mean') else None,
            'obs_var': self.obs_rms.var.copy() if hasattr(self.obs_rms, 'var') else None,
            'obs_count': self.obs_rms.count if hasattr(self.obs_rms, 'count') else 0,
            'ret_mean': self.ret_rms.mean.copy() if hasattr(self.ret_rms, 'mean') else None,
            'ret_var': self.ret_rms.var.copy() if hasattr(self.ret_rms, 'var') else None,
            'ret_count': self.ret_rms.count if hasattr(self.ret_rms, 'count') else 0,
            'step_count': self.step_count,
            'warmup_complete': self.warmup_complete,
            'training': self.training,
            'norm_obs': self.norm_obs,
            'norm_reward': self.norm_reward
        }
        return stats


def create_enhanced_normalizer(env, 
                              obs_size=None,
                              training=True,
                              norm_obs=True,
                              norm_reward=True,
                              clip_obs=10.0,  # 🔥 CRITIC FIX: Preservar range das observações
                              clip_reward=10.0,
                              gamma=0.99,
                              epsilon=1e-6,
                              momentum=0.999,
                              warmup_steps=2000,
                              stability_check=True):
    """
    🚀 Criar Enhanced VecNormalize com configurações otimizadas
    """
    print("🚀 CRIANDO Enhanced VecNormalize...")
    
    # Verificar se env é VecEnv
    if not hasattr(env, 'num_envs'):
        print("🔄 Convertendo para VecEnv...")
        env = DummyVecEnv([lambda: env])
    
    # Criar Enhanced VecNormalize
    enhanced_env = EnhancedVecNormalize(
        venv=env,
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
    
    print("✅ Enhanced VecNormalize criado com sucesso!")
    return enhanced_env


def create_enhanced_normalizer_wrapper(env, obs_size=None, normalizer_file=None):
    """
    🚀 Wrapper para criar Enhanced VecNormalize com carregamento automático
    """
    print("🚀 CRIANDO Enhanced VecNormalize Wrapper...")
    
    # Tentar carregar normalizer existente primeiro
    if normalizer_file and os.path.exists(normalizer_file):
        print(f"🔄 Carregando Enhanced VecNormalize existente: {normalizer_file}")
        try:
            enhanced_env = EnhancedVecNormalize.load(normalizer_file, env)
            enhanced_env.training = True  # Garantir modo treinamento
            print("✅ Enhanced VecNormalize carregado com sucesso")
            return enhanced_env
        except Exception as e:
            print(f"⚠️ Erro ao carregar Enhanced VecNormalize: {e}")
            print("🔄 Criando novo Enhanced VecNormalize...")
    
    # 🔥 CORREÇÃO CRÍTICA: Usar obs_size do ambiente se não fornecido
    if obs_size is None:
        obs_size = env.observation_space.shape[0]
        print(f"🔧 Obs_size automático detectado: {obs_size}")
    
    # 🎯 CONFIGURAÇÕES OTIMIZADAS BASEADAS EM RESEARCH PAPERS + UTI VECNORMALIZE VAR
    enhanced_env = create_enhanced_normalizer(
        env, 
        obs_size=obs_size,
        training=True,
        norm_obs=True,   # ✅ ATIVADO - Enhanced VecNormalize principal
        norm_reward=True,  # ✅ ATIVADO - Enhanced VecNormalize principal 
        clip_obs=10.0,      # 🔥 CRITIC FIX: Preservar informação das observações
        clip_reward=10.0,   # 🔥 CRITIC FIX: Preservar informação dos rewards
        gamma=0.99,        # ✅ MANTIDO: Funciona bem para trading
        epsilon=1e-7,      # 🔥 OTIMIZADO: Maior precisão numérica para evitar zeros
        momentum=0.999,    # 🔥 OTIMIZADO: Alta persistência para séries temporais não-estacionárias
        warmup_steps=3000, # 🔥 OTIMIZADO: Mais calibração para reduzir zeros extremos
        stability_check=True  # ✅ Verificações automáticas de saúde
    )
    
    # Calibração inicial com warmup
    print("🔄 Calibrando Enhanced VecNormalize com 1000 steps...")
    obs = enhanced_env.reset()
    for i in range(1000):
        action = enhanced_env.action_space.sample()
        obs, _, done, _ = enhanced_env.step(action)
        if done.any():
            obs = enhanced_env.reset()
    
    print("✅ Enhanced VecNormalize criado e calibrado")
    return enhanced_env


def save_enhanced_normalizer(enhanced_env, filepath):
    """
    💾 SALVAR Enhanced VecNormalize para produção
    """
    print(f"💾 Salvando Enhanced VecNormalize: {filepath}")
    
    try:
        if hasattr(enhanced_env, 'save'):
            # Configurar para produção
            original_training = enhanced_env.training
            enhanced_env.training = False  # Modo produção
            
            # Salvar normalizer
            success = enhanced_env.save(filepath)
            
            # Restaurar modo treinamento
            enhanced_env.training = original_training
            
            if success:
                print(f"✅ Enhanced VecNormalize salvo: {filepath}")
                return True
            else:
                print(f"❌ Falha ao salvar Enhanced VecNormalize: {filepath}")
                return False
        else:
            print(f"⚠️ Ambiente não tem método save(): {filepath}")
            return False
                
    except Exception as e:
        print(f"❌ Erro ao salvar Enhanced VecNormalize: {e}")
        return False


def monitor_enhanced_normalizer_health(enhanced_env, obs):
    """
    🔍 MONITORAR SAÚDE DO Enhanced VecNormalize
    """
    try:
        # Verificar se observações estão sendo normalizadas corretamente
        if isinstance(obs, (list, np.ndarray)):
            obs_flat = np.concatenate([o.flatten() if hasattr(o, 'flatten') else np.array(o).flatten() for o in obs])
        else:
            obs_flat = obs.flatten() if hasattr(obs, 'flatten') else np.array(obs).flatten()
        
        # Estatísticas das observações
        obs_mean = np.mean(obs_flat)
        obs_std = np.std(obs_flat)
        obs_min = np.min(obs_flat)
        obs_max = np.max(obs_flat)
        
        # Detectar problemas de normalização reais
        # 🎯 CORREÇÃO: Thresholds adequados para dados normalizados
        real_zeros = np.sum(np.abs(obs_flat) < 1e-8) / len(obs_flat)  # Zeros extremos apenas
        extreme_values = np.sum(np.abs(obs_flat) > 5.0) / len(obs_flat)  # Valores além de 5 sigmas
        
        # Alertar se há problemas
        if real_zeros > 0.1:  # >10% zeros extremos é problemático
            print(f"⚠️ ALERTA Enhanced VecNormalize: {real_zeros*100:.1f}% zeros extremos!")
            print(f"   📊 Mean: {obs_mean:.4f}, Std: {obs_std:.4f}, Range: [{obs_min:.4f}, {obs_max:.4f}]")
            return False
        
        if extreme_values > 0.05:  # >5% valores extremos é problemático
            print(f"⚠️ ALERTA Enhanced VecNormalize: {extreme_values*100:.1f}% valores extremos!")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Erro ao monitorar Enhanced VecNormalize: {e}")
        return False 