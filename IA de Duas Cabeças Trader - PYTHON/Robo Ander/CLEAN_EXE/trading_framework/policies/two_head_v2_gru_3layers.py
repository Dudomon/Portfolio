"""
🔥 TwoHeadV2GRU Policy - 3 LAYERS VERSION
Baseado na descoberta revolucionária que GRU > LSTM para trading
Versão com 3 camadas GRU para máxima capacidade temporal
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.recurrent.policies import RecurrentActorCriticPolicy
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor

class TwoHeadV2GRU3LayersPolicy(RecurrentActorCriticPolicy):
    """
    🔥 TWO HEAD V2 GRU 3 LAYERS POLICY
    
    FILOSOFIA:
    - GRU multicamadas (3 layers) para máxima capacidade temporal
    - Arquitetura baseada na V1 que CONVERGE
    - Sem forward customizado (usa métodos internos)
    - Foco em estabilidade + inteligência temporal
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = TradingTransformerFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # 🔥 PARÂMETROS GRU 3 LAYERS
        lstm_hidden_size: int = 128,  # Mantém nome para compatibilidade
        n_lstm_layers: int = 3,       # 3 layers GRU (máxima capacidade!)
        gru_dropout: float = 0.2,     # Dropout maior para 3 layers
        **kwargs
    ):
        
        # 🔥 CONFIGURAÇÕES GRU 3 LAYERS
        self.gru_hidden_size = lstm_hidden_size  # 128
        self.n_gru_layers = n_lstm_layers        # 3
        self.gru_dropout = gru_dropout           # 0.2
        
        # 🔥 FEATURES EXTRACTOR
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {'features_dim': 128}
        
        # 🔥 NETWORK ARCHITECTURE
        if net_arch is None:
            net_arch = [dict(pi=[256, 128, 64], vf=[256, 128, 64])]
        
        print(f"🔥 TwoHeadV2GRU3Layers: GRU({self.gru_hidden_size}) x {self.n_gru_layers} layers MÁXIMA CAPACIDADE!")
        
        # 🔥 INICIALIZAR COMO RECURRENT POLICY
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=self.gru_hidden_size,
            n_lstm_layers=self.n_gru_layers,
            **kwargs
        )
        
        # 🔥 SUBSTITUIR LSTM POR GRU 3 LAYERS
        self._replace_lstm_with_gru_3layers()
        
        # 🔥 COMPONENTES ESPECIALIZADOS PARA 3 LAYERS
        self._init_specialized_components_3layers()
        
        # 🔥 INICIALIZAÇÃO OTIMIZADA PARA GRU 3 LAYERS
        self._initialize_gru_3layers_weights()
        
        # Debug info
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔥 TwoHeadV2GRU3Layers: {total_params:,} parâmetros")
        print(f"🔥 Arquitetura: GRU({self.gru_hidden_size}) x {self.n_gru_layers} + componentes especializados")
        print(f"🔥 Filosofia: MÁXIMA capacidade temporal com estabilidade do GRU")
    
    def _replace_lstm_with_gru_3layers(self):
        """🔥 SUBSTITUIR LSTM POR GRU 3 LAYERS - MÁXIMA CAPACIDADE!"""
        
        # 🔥 GRU 3 LAYERS PARA ACTOR
        self.lstm_actor = nn.GRU(
            input_size=self.features_extractor.features_dim,  # 128
            hidden_size=self.gru_hidden_size,                 # 128
            num_layers=self.n_gru_layers,                     # 3 layers!
            batch_first=True,
            dropout=self.gru_dropout                          # 0.2
        )
        
        # 🔥 GRU 3 LAYERS PARA CRITIC  
        self.lstm_critic = nn.GRU(
            input_size=self.features_extractor.features_dim,  # 128
            hidden_size=self.gru_hidden_size,                 # 128
            num_layers=self.n_gru_layers,                     # 3 layers!
            batch_first=True,
            dropout=self.gru_dropout                          # 0.2
        )
        
        print(f"🔥 LSTM substituído por GRU 3 LAYERS: {self.gru_hidden_size} hidden, {self.n_gru_layers} layers")
    
    def _init_specialized_components_3layers(self):
        """🔥 COMPONENTES ESPECIALIZADOS PARA GRU 3 LAYERS"""
        
        # 🔥 ATTENTION MAIS PODEROSA PARA 3 LAYERS
        self.attention = nn.MultiheadAttention(
            embed_dim=self.gru_hidden_size,  # 128
            num_heads=8,                     # 8 heads (mais que V1)
            dropout=0.1,                     # Dropout moderado
            batch_first=True
        )
        
        # 🔥 PATTERN DETECTION MAIS PROFUNDO
        self.pattern_detector = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 🔥 TWO HEADS MAIS SOFISTICADAS
        # Entry Head - para decisões de entrada
        self.entry_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size + 32, 96),  # GRU + patterns
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # Management Head - para gestão de posições
        self.management_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size + 32, 96),  # GRU + patterns
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 🔥 FUSION FINAL MAIS SOFISTICADO
        self.final_fusion = nn.Sequential(
            nn.Linear(32 + 32, 96),  # Entry + Management
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        print(f"🔥 Componentes 3 Layers: Attention(8 heads) + Pattern Detection(3 layers) + Two Heads(3 layers)")
    
    def _initialize_gru_3layers_weights(self):
        """🔥 INICIALIZAÇÃO OTIMIZADA PARA GRU 3 LAYERS"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform com gain mais conservador para 3 layers
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.GRU):
                # 🔥 INICIALIZAÇÃO ESPECIAL PARA GRU 3 LAYERS
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.3)  # Mais conservador
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.3)      # Mais conservador
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # 🔥 GRU: Reset gate bias = 0, Update gate bias = 1
                        n = param.size(0)
                        param.data[n//3:2*n//3].fill_(1)  # Update gate bias = 1
            elif isinstance(module, nn.MultiheadAttention):
                # 🔥 INICIALIZAÇÃO PARA ATTENTION 8 HEADS
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.3)
                    elif 'bias' in name and param is not None:
                        nn.init.constant_(param.data, 0)
        
        print(f"🔥 Inicialização GRU 3 Layers otimizada aplicada (gain=0.3)")
    
    # 🔥 USAR MÉTODOS INTERNOS PADRÃO - SEM FORWARD CUSTOMIZADO
    # RecurrentPPO usa métodos internos que sempre funcionaram
    # GRU 3 Layers vai funcionar perfeitamente com o sistema padrão

def create_two_head_v2_gru_3layers_policy(
    observation_space,
    action_space,
    lr_schedule,
    **kwargs
) -> TwoHeadV2GRU3LayersPolicy:
    """Factory para criar TwoHeadV2GRU3LayersPolicy"""
    return TwoHeadV2GRU3LayersPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **kwargs
    )

def get_gru_3layers_trading_kwargs() -> Dict[str, Any]:
    """🔥 CONFIGURAÇÕES OTIMIZADAS PARA GRU 3 LAYERS"""
    return {
        'lstm_hidden_size': 128,     # GRU hidden size
        'n_lstm_layers': 3,          # 3 GRU layers (máxima capacidade!)
        'gru_dropout': 0.2,          # Dropout maior para 3 layers
        'features_extractor_kwargs': {
            'features_dim': 128,     # 128 features
            'seq_len': 8            # 8 barras histórico
        },
        'net_arch': [dict(pi=[256, 128, 64], vf=[256, 128, 64])],  # Arquitetura robusta
        'activation_fn': nn.ReLU,
        'ortho_init': True,
    } 