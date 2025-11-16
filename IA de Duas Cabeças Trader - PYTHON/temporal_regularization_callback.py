#!/usr/bin/env python3
"""
🚀 Temporal Regularization Callback
Garante que a loss de regularização temporal é aplicada durante o treinamento
"""

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

class TemporalRegularizationCallback(BaseCallback):
    """
    🚀 Callback que aplica temporal regularization loss durante treinamento
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.temporal_losses = []
        
    def _on_step(self) -> bool:
        """Chamado a cada step"""
        return True
    
    def _on_rollout_end(self) -> bool:
        """Chamado no final de cada rollout"""
        
        # Verificar se policy tem temporal regularization
        if hasattr(self.model.policy, 'get_temporal_regularization_loss'):
            try:
                # Obter loss de regularização temporal
                temporal_loss = self.model.policy.get_temporal_regularization_loss()
                
                if isinstance(temporal_loss, torch.Tensor):
                    temporal_loss_value = temporal_loss.item()
                else:
                    temporal_loss_value = float(temporal_loss)
                
                # Armazenar para logging
                self.temporal_losses.append(temporal_loss_value)
                
                # Log para TensorBoard se disponível
                for output_format in self.logger.output_formats:
                    if isinstance(output_format, TensorBoardOutputFormat):
                        output_format.writer.add_scalar(
                            'train/temporal_regularization_loss', 
                            temporal_loss_value, 
                            self.num_timesteps
                        )
                
                # Print periódico
                if len(self.temporal_losses) % 100 == 0:
                    avg_temp_loss = np.mean(self.temporal_losses[-100:])
                    print(f"🚀 Temporal Regularization - Avg Loss (últimos 100): {avg_temp_loss:.6f}")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠️ Erro ao aplicar temporal regularization: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Chamado no final do treinamento"""
        if self.temporal_losses:
            total_avg = np.mean(self.temporal_losses)
            print(f"🚀 Temporal Regularization - Loss média total: {total_avg:.6f}")
            print(f"🚀 Total de aplicações: {len(self.temporal_losses)}")