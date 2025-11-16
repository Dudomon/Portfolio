"""
🔥 OPUS: CURIOSITY-DRIVEN REWARDS
Sistema que recompensa exploração de estados não vistos anteriormente
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CuriosityRewardCalculator:
    """
    Sistema de curiosity rewards para incentivar exploration
    Baseado em prediction error de um forward model
    """
    
    def __init__(self, 
                 state_dim: int = 2580,
                 action_dim: int = 11,
                 feature_dim: int = 128,
                 learning_rate: float = 1e-4,
                 curiosity_weight: float = 0.01,
                 device: str = 'cuda'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.curiosity_weight = curiosity_weight
        self.device = device
        
        # Feature extractor - encode states into compact features
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim)
        ).to(device)
        
        # Forward model - predict next state features from current + action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + 
            list(self.forward_model.parameters()),
            lr=learning_rate
        )
        
        # História de estados para detecção de novelty
        self.state_history = deque(maxlen=10000)
        self.prediction_errors = deque(maxlen=1000)
        
        # Normalizadores
        self.state_normalizer = self._create_normalizer()
        self.update_count = 0
        
    def _create_normalizer(self):
        """Cria normalizador robusto para estados"""
        return {
            'mean': torch.zeros(self.state_dim, device=self.device),
            'std': torch.ones(self.state_dim, device=self.device),
            'count': 0
        }
    
    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normaliza estado usando running statistics"""
        if self.state_normalizer['count'] < 100:
            # Bootstrap: usar normalização simples
            return (state - state.mean()) / (state.std() + 1e-8)
        
        # Usar running statistics
        return (state - self.state_normalizer['mean']) / (self.state_normalizer['std'] + 1e-8)
    
    def _update_normalizer(self, state: torch.Tensor):
        """Atualiza running statistics do normalizador"""
        count = self.state_normalizer['count']
        
        if count == 0:
            self.state_normalizer['mean'] = state.mean(dim=0)
            self.state_normalizer['std'] = state.std(dim=0)
        else:
            # Running average
            alpha = 0.001  # Taxa de atualização lenta
            self.state_normalizer['mean'] = (1 - alpha) * self.state_normalizer['mean'] + alpha * state.mean(dim=0)
            self.state_normalizer['std'] = (1 - alpha) * self.state_normalizer['std'] + alpha * state.std(dim=0)
        
        self.state_normalizer['count'] += 1
    
    def compute_intrinsic_reward(self, 
                                state: np.ndarray, 
                                action: np.ndarray, 
                                next_state: np.ndarray) -> float:
        """
        Computa reward intrínseco baseado em prediction error
        
        Args:
            state: Estado atual
            action: Ação tomada  
            next_state: Próximo estado
            
        Returns:
            Reward intrínseco (maior = mais novel/interessante)
        """
        try:
            # Convert to tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 🔥 FIX: Ensure action has correct dimensions (pad if necessary)
            action = np.array(action).flatten()
            if len(action) < self.action_dim:
                # Pad with zeros to match expected action_dim
                padded_action = np.zeros(self.action_dim)
                padded_action[:len(action)] = action
                action = padded_action
            elif len(action) > self.action_dim:
                # Truncate if too long
                action = action[:self.action_dim]
            
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Update normalizer
            self._update_normalizer(state_tensor)
            
            # Normalize states
            state_norm = self._normalize_state(state_tensor)
            next_state_norm = self._normalize_state(next_state_tensor)
            
            with torch.no_grad():
                # Extract features
                state_features = self.feature_extractor(state_norm)
                next_state_features = self.feature_extractor(next_state_norm)
                
                # Predict next state features
                forward_input = torch.cat([state_features, action_tensor], dim=-1)
                predicted_features = self.forward_model(forward_input)
                
                # Calculate prediction error (novelty measure)
                prediction_error = F.mse_loss(predicted_features, next_state_features, reduction='mean')
                
            # Train the models if we have enough experience
            if len(self.state_history) > 32 and self.update_count % 4 == 0:
                self._update_models(state_norm, action_tensor, next_state_norm)
            
            # Store experience
            self.state_history.append(state.copy())
            prediction_error_value = prediction_error.item()
            self.prediction_errors.append(prediction_error_value)
            
            # Normalize reward by recent prediction errors
            if len(self.prediction_errors) >= 100:
                recent_errors = list(self.prediction_errors)[-100:]
                mean_error = np.mean(recent_errors)
                std_error = np.std(recent_errors) + 1e-8
                
                # Z-score normalization with clipping
                normalized_error = (prediction_error_value - mean_error) / std_error
                normalized_error = np.clip(normalized_error, -3, 3)  # Clip outliers
                
                # Convert to positive reward
                intrinsic_reward = self.curiosity_weight * max(0, normalized_error)
            else:
                # Bootstrap phase
                intrinsic_reward = self.curiosity_weight * prediction_error_value
            
            self.update_count += 1
            
            # Log ocasionalmente
            if self.update_count % 1000 == 0:
                avg_error = np.mean(list(self.prediction_errors)[-100:]) if self.prediction_errors else 0
                logger.info(f"[CURIOSITY] Step {self.update_count}: avg_pred_error={avg_error:.4f}, "
                          f"intrinsic_reward={intrinsic_reward:.4f}")
            
            return float(intrinsic_reward)
            
        except Exception as e:
            logger.warning(f"[CURIOSITY] Error computing intrinsic reward: {e}")
            return 0.0
    
    def _update_models(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        """Atualiza os modelos de curiosidade"""
        try:
            self.optimizer.zero_grad()
            
            # Extract features
            state_features = self.feature_extractor(state)
            next_state_features = self.feature_extractor(next_state)
            
            # 🔥 FIX: Ensure action tensor has correct shape for concatenation
            if action.size(-1) != self.action_dim:
                # Reshape or pad action to match expected dimensions
                if action.numel() < self.action_dim:
                    padded_action = torch.zeros(action.size(0), self.action_dim, device=action.device)
                    padded_action[:, :action.size(-1)] = action
                    action = padded_action
                else:
                    action = action[:, :self.action_dim]
            
            # Forward model prediction
            forward_input = torch.cat([state_features, action], dim=-1)
            predicted_features = self.forward_model(forward_input)
            
            # Loss: prediction error
            forward_loss = F.mse_loss(predicted_features, next_state_features.detach())
            
            # Feature loss: encourage informative features
            feature_loss = -0.1 * torch.mean(torch.std(state_features, dim=0))
            
            total_loss = forward_loss + feature_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.feature_extractor.parameters()) + 
                list(self.forward_model.parameters()), 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
        except Exception as e:
            logger.warning(f"[CURIOSITY] Error updating models: {e}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Retorna estatísticas do sistema de curiosidade"""
        if not self.prediction_errors:
            return {}
        
        recent_errors = list(self.prediction_errors)[-100:]
        
        return {
            'curiosity_avg_error': np.mean(recent_errors),
            'curiosity_std_error': np.std(recent_errors),
            'curiosity_max_error': np.max(recent_errors),
            'curiosity_update_count': self.update_count,
            'curiosity_weight': self.curiosity_weight
        }
    
    def save_state(self, path: str):
        """Salva o estado dos modelos"""
        try:
            torch.save({
                'feature_extractor': self.feature_extractor.state_dict(),
                'forward_model': self.forward_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'state_normalizer': self.state_normalizer,
                'update_count': self.update_count
            }, path)
            logger.info(f"[CURIOSITY] State saved to {path}")
        except Exception as e:
            logger.warning(f"[CURIOSITY] Error saving state: {e}")
    
    def load_state(self, path: str):
        """Carrega o estado dos modelos"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            self.forward_model.load_state_dict(checkpoint['forward_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.state_normalizer = checkpoint['state_normalizer']
            self.update_count = checkpoint['update_count']
            
            logger.info(f"[CURIOSITY] State loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"[CURIOSITY] Error loading state: {e}")
            return False