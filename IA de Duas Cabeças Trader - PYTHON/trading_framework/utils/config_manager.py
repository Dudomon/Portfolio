#!/usr/bin/env python3
"""
⚙️ MÓDULO DE GERENCIAMENTO DE CONFIGURAÇÕES INDEPENDENTE
Sistema de configurações centralizado e reutilizável
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging


@dataclass
class TrainingConfig:
    """📚 Configurações de treinamento"""
    # PPO Parameters
    learning_rate: float = 2.5e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    
    # Training Control
    total_timesteps: int = 1000000
    save_freq: int = 10000
    eval_freq: int = 50000
    log_interval: int = 100
    
    # Environment
    window_size: int = 20
    initial_balance: float = 1000.0
    transaction_fee: float = 0.001
    max_position_size: float = 1.0
    
    # Trading Parameters
    target_trades_per_day: int = 16
    min_trade_duration: int = 3
    max_trade_duration: int = 100
    
    # Risk Management
    max_drawdown_tolerance: float = 0.2
    stop_loss_range: tuple = (0.01, 0.05)
    take_profit_range: tuple = (0.02, 0.10)


@dataclass
class ModelConfig:
    """🧠 Configurações do modelo"""
    # Architecture
    policy_type: str = "TwoHeadPolicy"
    features_extractor_type: str = "TransformerFeatureExtractor"
    features_dim: int = 128
    
    # Transformer
    n_heads: int = 8
    n_layers: int = 4
    d_model: int = 128
    dropout: float = 0.1
    
    # LSTM
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    
    # MLP
    mlp_hidden_sizes: tuple = (256, 128, 64)
    mlp_dropout: float = 0.1


@dataclass
class RewardConfig:
    """🎯 Configurações do sistema de recompensas"""
    # Weights
    portfolio_weight: float = 0.7
    drawdown_weight: float = 0.2
    activity_weight: float = 0.1
    
    # Penalties
    flip_flop_penalty: float = -8.0
    micro_trade_penalty: float = -5.0
    hold_penalty: float = -0.1
    
    # Bonuses
    win_bonus: float = 2.0
    target_zone_bonus: float = 0.1
    activity_bonus: float = 0.2
    
    # Anti-overtrading
    overtrading_threshold: int = 50
    overtrading_penalty: float = -50.0
    undertrading_penalty: float = -15.0


@dataclass
class DataConfig:
    """📊 Configurações de dados"""
    # Data Loading
    data_dir: str = "data"
    cache_dir: str = "data_cache"
    use_cache: bool = True
    
    # Processing
    remove_outliers: bool = True
    outlier_percentile: float = 0.999
    add_technical_features: bool = True
    
    # Validation
    required_columns: tuple = ('timestamp', 'close_5m', 'volume_5m')
    min_data_points: int = 1000


@dataclass
class SystemConfig:
    """💻 Configurações do sistema"""
    # Hardware
    device: str = "auto"  # "cpu", "cuda", "auto"
    num_threads: int = 4
    use_vecnormalize: bool = False  # 🚨 DESABILITADO - USAMOS ENHANCED SCALER
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    tensorboard_log: str = "tensorboard_logs"
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    backup_locations: tuple = ("models", "trading_framework/training/checkpoints")
    
    # Performance
    enable_profiling: bool = False
    memory_efficient: bool = True


class ConfigManager:
    """⚙️ Gerenciador central de configurações"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.training_config = TrainingConfig()
        self.model_config = ModelConfig()
        self.reward_config = RewardConfig()
        self.data_config = DataConfig()
        self.system_config = SystemConfig()
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """Carrega configurações de arquivo"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            print(f"⚠️ Arquivo de configuração não encontrado: {config_file}")
            return
        
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {config_path.suffix}")
            
            self._update_configs(config_data)
            print(f"✅ Configurações carregadas: {config_file}")
            
        except Exception as e:
            print(f"⚠️ Erro ao carregar configurações: {e}")
    
    def save_config(self, config_file: str) -> None:
        """Salva configurações em arquivo"""
        try:
            config_data = self.get_all_configs()
            
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {config_path.suffix}")
            
            print(f"💾 Configurações salvas: {config_file}")
            
        except Exception as e:
            print(f"⚠️ Erro ao salvar configurações: {e}")
    
    def _update_configs(self, config_data: Dict[str, Any]) -> None:
        """Atualiza configurações com dados carregados"""
        if 'training' in config_data:
            self._update_dataclass(self.training_config, config_data['training'])
        
        if 'model' in config_data:
            self._update_dataclass(self.model_config, config_data['model'])
        
        if 'reward' in config_data:
            self._update_dataclass(self.reward_config, config_data['reward'])
        
        if 'data' in config_data:
            self._update_dataclass(self.data_config, config_data['data'])
        
        if 'system' in config_data:
            self._update_dataclass(self.system_config, config_data['system'])
    
    def _update_dataclass(self, dataclass_instance, data: Dict[str, Any]) -> None:
        """Atualiza dataclass com dados"""
        for key, value in data.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Retorna todas as configurações como dict"""
        return {
            'training': asdict(self.training_config),
            'model': asdict(self.model_config),
            'reward': asdict(self.reward_config),
            'data': asdict(self.data_config),
            'system': asdict(self.system_config)
        }
    
    def get_ppo_params(self) -> Dict[str, Any]:
        """Retorna parâmetros para PPO"""
        return {
            'learning_rate': self.training_config.learning_rate,
            'n_steps': self.training_config.n_steps,
            'batch_size': self.training_config.batch_size,
            'n_epochs': self.training_config.n_epochs,
            'gamma': self.training_config.gamma,
            'gae_lambda': self.training_config.gae_lambda,
            'clip_range': self.training_config.clip_range,
            'clip_range_vf': self.training_config.clip_range_vf,
            'ent_coef': self.training_config.ent_coef,
            'vf_coef': self.training_config.vf_coef,
            'max_grad_norm': self.training_config.max_grad_norm,
            'target_kl': self.training_config.target_kl,
            'tensorboard_log': self.system_config.tensorboard_log,
            'verbose': 1
        }
    
    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Retorna parâmetros para política"""
        return {
            'features_extractor_type': self.model_config.features_extractor_type,
            'features_dim': self.model_config.features_dim,
            'n_heads': self.model_config.n_heads,
            'n_layers': self.model_config.n_layers,
            'd_model': self.model_config.d_model,
            'dropout': self.model_config.dropout,
            'lstm_hidden_size': self.model_config.lstm_hidden_size,
            'lstm_layers': self.model_config.lstm_layers,
            'lstm_dropout': self.model_config.lstm_dropout,
            'mlp_hidden_sizes': self.model_config.mlp_hidden_sizes,
            'mlp_dropout': self.model_config.mlp_dropout
        }
    
    def print_config_summary(self) -> None:
        """Imprime resumo das configurações"""
        print("⚙️ RESUMO DAS CONFIGURAÇÕES")
        print("=" * 50)
        
        print(f"📚 TREINAMENTO:")
        print(f"   Learning Rate: {self.training_config.learning_rate:.2e}")
        print(f"   Total Steps: {self.training_config.total_timesteps:,}")
        print(f"   Batch Size: {self.training_config.batch_size}")
        print(f"   Target Trades/Day: {self.training_config.target_trades_per_day}")
        
        print(f"\n🧠 MODELO:")
        print(f"   Policy: {self.model_config.policy_type}")
        print(f"   Features Extractor: {self.model_config.features_extractor_type}")
        print(f"   Features Dim: {self.model_config.features_dim}")
        
        print(f"\n🎯 REWARDS:")
        print(f"   Portfolio Weight: {self.reward_config.portfolio_weight}")
        print(f"   Drawdown Weight: {self.reward_config.drawdown_weight}")
        print(f"   Activity Weight: {self.reward_config.activity_weight}")
        
        print(f"\n💻 SISTEMA:")
        print(f"   Device: {self.system_config.device}")
        print(f"   Use VecNormalize: {self.system_config.use_vecnormalize}")
        print(f"   Memory Efficient: {self.system_config.memory_efficient}")
        
        print("=" * 50)
    
    def validate_configs(self) -> Dict[str, Any]:
        """Valida configurações e retorna relatório"""
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validar learning rate
        if self.training_config.learning_rate <= 0:
            report['is_valid'] = False
            report['errors'].append("Learning rate deve ser > 0")
        
        # Validar batch size
        if self.training_config.batch_size <= 0:
            report['is_valid'] = False
            report['errors'].append("Batch size deve ser > 0")
        
        # Validar n_steps
        if self.training_config.n_steps <= 0:
            report['is_valid'] = False
            report['errors'].append("n_steps deve ser > 0")
        
        # Validar initial balance
        if self.training_config.initial_balance <= 0:
            report['is_valid'] = False
            report['errors'].append("Initial balance deve ser > 0")
        
        # Avisos
        if self.training_config.learning_rate > 1e-2:
            report['warnings'].append("Learning rate muito alto (> 1e-2)")
        
        if self.training_config.batch_size > 512:
            report['warnings'].append("Batch size muito alto (> 512)")
        
        return report


# Configurações predefinidas
PRESET_CONFIGS = {
    'conservative': {
        'training': {
            'learning_rate': 1e-4,
            'target_trades_per_day': 8,
            'max_drawdown_tolerance': 0.15
        },
        'reward': {
            'portfolio_weight': 0.8,
            'drawdown_weight': 0.3,
            'activity_weight': 0.05
        }
    },
    'aggressive': {
        'training': {
            'learning_rate': 5e-4,
            'target_trades_per_day': 25,
            'max_drawdown_tolerance': 0.3
        },
        'reward': {
            'portfolio_weight': 0.6,
            'drawdown_weight': 0.1,
            'activity_weight': 0.2
        }
    },
    'balanced': {
        'training': {
            'learning_rate': 2.5e-4,
            'target_trades_per_day': 16,
            'max_drawdown_tolerance': 0.2
        },
        'reward': {
            'portfolio_weight': 0.7,
            'drawdown_weight': 0.2,
            'activity_weight': 0.1
        }
    }
}


def load_preset_config(preset_name: str) -> ConfigManager:
    """Carrega configuração predefinida"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Preset '{preset_name}' não encontrado. Disponíveis: {list(PRESET_CONFIGS.keys())}")
    
    config_manager = ConfigManager()
    config_manager._update_configs(PRESET_CONFIGS[preset_name])
    
    return config_manager 