"""
Feature Extractors - Módulo de Extração de Features
==================================================

Este módulo contém implementações de feature extractors customizados
para processamento de dados de mercado financeiro.

Extractors disponíveis:
- TransformerFeatureExtractor: Extrator baseado em Transformer
"""

from .transformer_extractor import TransformerFeatureExtractor

__all__ = [
    'TransformerFeatureExtractor'
] 