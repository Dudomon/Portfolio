"""
🔥 TRADING FRAMEWORK POLICIES

Políticas especializadas para trading algorítmico:
- TwoHeadPolicy: Política com duas cabeças (estratégica e tática)
- TwoHeadV2Policy: Política otimizada para trading 48h (nova versão)
- Funções auxiliares para criação e configuração
"""

from .two_head_policy import TwoHeadPolicy, create_two_head_policy, get_default_policy_kwargs
from .two_head_v2 import TwoHeadV2Policy, create_two_head_v2_policy, get_optimized_trading_kwargs

__all__ = [
    'TwoHeadPolicy',
    'create_two_head_policy',
    'get_default_policy_kwargs',
    'TwoHeadV2Policy',
    'create_two_head_v2_policy', 
    'get_optimized_trading_kwargs',
] 