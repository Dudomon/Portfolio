#!/usr/bin/env python3
"""
🔄 MÓDULO DE CALLBACKS INDEPENDENTE
Sistema de callbacks para treinamento PPO
"""

import os
import json
import zipfile
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from pathlib import Path
import logging


class RobustSaveCallback(BaseCallback):
    """💾 Callback de salvamento robusto com múltiplos locais"""
    
    def __init__(self, save_freq: int = 10000, save_path: str = "models", 
                 name_prefix: str = "ppo_model", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Múltiplos locais de salvamento
        self.backup_paths = [
            self.save_path,
            Path("trading_framework/training/checkpoints"),
            Path("treino_principal/models")
        ]
        
        for path in self.backup_paths:
            path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        return True
    
    def _save_model(self):
        """Salva modelo com verificação de integridade"""
        try:
            # Salvar modelo principal
            model_path = self.save_path / f"{self.name_prefix}_{self.n_calls}_steps"
            self.model.save(str(model_path))
            
            # Salvar estado do ambiente
            self._save_environment_state(model_path)
            
            # Backup em múltiplos locais
            for backup_path in self.backup_paths[1:]:
                backup_model_path = backup_path / f"{self.name_prefix}_{self.n_calls}_steps"
                self._copy_model_files(model_path, backup_model_path)
            
            if self.verbose > 0:
                print(f"💾 Modelo salvo: {model_path}")
                
        except Exception as e:
            print(f"⚠️ Erro ao salvar modelo: {e}")
    
    def _save_environment_state(self, model_path: Path):
        """Salva estado do ambiente"""
        try:
            env_state = {
                'current_step': self.n_calls,
                'portfolio_value': getattr(self.training_env, 'portfolio_value', 0.0),
                'positions': getattr(self.training_env, 'positions', []),
                'trades': getattr(self.training_env, 'trades', []),
                'drawdown': getattr(self.training_env, 'drawdown', 0.0),
                'peak_portfolio': getattr(self.training_env, 'peak_portfolio', 0.0)
            }
            
            state_file = model_path / "environment_state.json"
            with open(state_file, 'w') as f:
                json.dump(env_state, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Erro ao salvar estado do ambiente: {e}")
    
    def _copy_model_files(self, source_path: Path, dest_path: Path):
        """Copia arquivos do modelo para backup"""
        try:
            if source_path.exists():
                # Criar ZIP com todos os arquivos
                zip_path = dest_path.with_suffix('.zip')
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in source_path.rglob('*'):
                        if file_path.is_file():
                            zipf.write(file_path, file_path.relative_to(source_path))
                            
        except Exception as e:
            print(f"⚠️ Erro ao fazer backup: {e}")


class LearningMonitorCallback(BaseCallback):
    """📊 Monitor de aprendizado com detecção de problemas"""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.loss_history = []
        self.lr_history = []
        self.gradient_history = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._monitor_learning()
        return True
    
    def _monitor_learning(self):
        """Monitora métricas de aprendizado"""
        try:
            # Capturar loss atual
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                loss = self.model.logger.name_to_value.get('train/loss', 0.0)
                self.loss_history.append(loss)
            
            # Capturar learning rate
            lr = self._get_learning_rate()
            self.lr_history.append(lr)
            
            # Capturar gradientes
            gradients = self._get_gradients()
            self.gradient_history.append(gradients)
            
            # Verificar problemas
            self._check_learning_issues()
            
        except Exception as e:
            print(f"⚠️ Erro no monitor de aprendizado: {e}")
    
    def _get_learning_rate(self) -> float:
        """Captura learning rate atual"""
        try:
            # Método 1: Via optimizer
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                for param_group in self.model.policy.optimizer.param_groups:
                    return param_group['lr']
            
            # Método 2: Via lr_schedule
            if hasattr(self.model, 'lr_schedule'):
                return self.model.lr_schedule(1.0)  # Progresso 100%
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_gradients(self) -> float:
        """Captura norma L2 dos gradientes"""
        try:
            total_norm = 0.0
            param_count = 0
            
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                return total_norm
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _check_learning_issues(self):
        """Verifica problemas de aprendizado"""
        if len(self.loss_history) < 2:
            return
        
        current_loss = self.loss_history[-1]
        current_lr = self.lr_history[-1]
        current_grad = self.gradient_history[-1]
        
        # Verificar loss muito alta
        if current_loss > 0.1:
            print(f"⚠️ Loss alta detectada: {current_loss:.4f}")
        
        # Verificar learning rate muito baixo
        if current_lr < 1e-6:
            print(f"⚠️ Learning rate muito baixo: {current_lr:.2e}")
        
        # Verificar gradientes muito baixos
        if current_grad < 1e-8:
            print(f"⚠️ Gradientes muito baixos: {current_grad:.2e}")
        
        # Verificar gradientes muito altos
        if current_grad > 10.0:
            print(f"⚠️ Gradientes muito altos: {current_grad:.2e}")


class VecNormalizeCallback(BaseCallback):
    """📊 Callback para gerenciar VecNormalize"""
    
    def __init__(self, save_freq: int = 10000, save_path: str = "models", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save_vecnormalize()
        return True
    
    def _save_vecnormalize(self):
        """Salva VecNormalize com sistema seguro"""
        try:
            if isinstance(self.training_env, VecNormalize):
                vecnorm_path = self.save_path / f"vecnormalize_{self.n_calls}_steps.pkl"
                self.training_env.save(str(vecnorm_path))
                
                if self.verbose > 0:
                    print(f"📊 VecNormalize salvo: {vecnorm_path}")
                    
        except Exception as e:
            print(f"⚠️ Erro ao salvar VecNormalize: {e}")
    
    @staticmethod
    def load_vecnormalize_safe(vecnorm_path: str, env) -> VecNormalize:
        """Carrega VecNormalize com sistema seguro"""
        try:
            if os.path.exists(vecnorm_path):
                vecnorm = VecNormalize.load(vecnorm_path, env)
                print("📊 VecNormalize carregado preservando estatísticas acumuladas")
                return vecnorm
            else:
                return VecNormalize(env)
                
        except Exception as e:
            print(f"⚠️ Erro ao carregar VecNormalize: {e}")
            return VecNormalize(env)


class EvaluationCallback(BaseCallback):
    """🎯 Callback para avaliação periódica"""
    
    def __init__(self, eval_freq: int = 50000, eval_episodes: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_results = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True
    
    def _run_evaluation(self):
        """Executa avaliação do modelo"""
        try:
            # Criar ambiente de avaliação
            eval_env = self._create_eval_env()
            
            # Executar avaliação
            results = self._evaluate_model(eval_env)
            
            # Salvar resultados
            self.eval_results.append({
                'step': self.n_calls,
                'results': results
            })
            
            # Log dos resultados
            if self.verbose > 0:
                self._log_evaluation_results(results)
                
        except Exception as e:
            print(f"⚠️ Erro na avaliação: {e}")
    
    def _create_eval_env(self):
        """Cria ambiente de avaliação"""
        # Implementar criação do ambiente de avaliação
        # Por enquanto, retorna None
        return None
    
    def _evaluate_model(self, eval_env) -> Dict[str, float]:
        """Avalia modelo e retorna métricas"""
        # Implementar avaliação
        # Por enquanto, retorna métricas vazias
        return {
            'portfolio_value': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def _log_evaluation_results(self, results: Dict[str, float]):
        """Log dos resultados de avaliação"""
        print(f"\n🎯 AVALIAÇÃO - Step {self.n_calls:,}")
        print(f"   💰 Portfolio: ${results.get('portfolio_value', 0):.2f}")
        print(f"   📊 Trades: {results.get('total_trades', 0)}")
        print(f"   🏆 Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"   📈 Sharpe: {results.get('sharpe_ratio', 0):.2f}")


class KeyboardEvaluationCallback(BaseCallback):
    """⌨️ Callback para avaliação via teclado"""
    
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.evaluation_queue = []
        self.keyboard_thread = None
        
    def _on_step(self) -> bool:
        # Verificar se há avaliações na fila
        if self.evaluation_queue:
            self._process_evaluation_queue()
        return True
    
    def _process_evaluation_queue(self):
        """Processa fila de avaliações"""
        while self.evaluation_queue:
            eval_request = self.evaluation_queue.pop(0)
            self._run_keyboard_evaluation(eval_request)
    
    def _run_keyboard_evaluation(self, eval_request: Dict[str, Any]):
        """Executa avaliação solicitada via teclado"""
        try:
            print(f"\n🎯 AVALIAÇÃO SOLICITADA - Step {self.n_calls:,}")
            
            # Implementar avaliação
            # Por enquanto, apenas log
            print("   📊 Avaliação em andamento...")
            
        except Exception as e:
            print(f"⚠️ Erro na avaliação via teclado: {e}")
    
    def add_evaluation_request(self, request: Dict[str, Any]):
        """Adiciona solicitação de avaliação à fila"""
        self.evaluation_queue.append(request) 