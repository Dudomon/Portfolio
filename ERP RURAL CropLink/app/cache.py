#!/usr/bin/env python3
"""
Sistema de Cache Hierárquico L2 - CropLink
Módulo dedicado para gerenciamento de cache multi-tenant com Flask-Caching

Funcionalidades:
- Cache L2 com Redis/SimpleCache
- Namespace por tenant (user_id)
- TTL inteligente baseado na volatilidade dos dados  
- Decorators para cache automático
- Invalidação por eventos
"""

import logging
from functools import wraps
from flask import current_app, g
from flask_login import current_user
from typing import Optional, Dict, Any, Callable, Union
import json
import hashlib

logger = logging.getLogger(__name__)


class CacheManager:
    """Gerenciador centralizado do sistema de cache L2"""
    
    # TTL baseado na volatilidade dos dados (segundos)
    TTL_CONFIG = {
        'dashboard_stats': 60,      # 1 minuto - dados dinâmicos
        'silo_occupancy': 30,       # 30 segundos - muito dinâmico  
        'grain_stocks': 45,         # 45 segundos - mudanças frequentes
        'supply_levels': 120,       # 2 minutos - mudanças moderadas
        'user_info': 300,           # 5 minutos - dados relativamente estáticos
        'machinery_list': 600,      # 10 minutos - raramente muda
        'client_list': 900,         # 15 minutos - mudanças esporádicas
        'reports': 180,             # 3 minutos - relatórios
        'rain_recent': 90,          # 1.5 minutos - dados meteorológicos
        'default': 300              # 5 minutos padrão
    }

    @staticmethod
    def get_cache():
        """Obter instância do cache Flask-Caching"""
        try:
            # Primeiro tenta obter do contexto da aplicação
            if hasattr(current_app, 'cache'):
                return current_app.cache
            
            # Depois tenta das extensões
            cache_ext = current_app.extensions.get('cache')
            if cache_ext and hasattr(cache_ext, 'set'):
                return cache_ext
                
            # Se não encontrar cache válido, retorna None (modo fallback)
            logger.warning("Cache não encontrado ou mal configurado - usando fallback")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter cache: {str(e)}")
            return None

    @staticmethod
    def get_tenant_key(base_key: str, user_id: Optional[int] = None, 
                      additional_params: Dict[str, Any] = None) -> str:
        """
        Gerar chave de cache com namespace por tenant
        
        Args:
            base_key: Chave base do cache
            user_id: ID do usuário/tenant (usa current_user se None)
            additional_params: Parâmetros adicionais para a chave
            
        Returns:
            Chave de cache formatada com namespace
        """
        # Usar current_user se user_id não fornecido
        if user_id is None and current_user.is_authenticated:
            user_id = current_user.id
        
        # Chave base com tenant
        cache_key = f"tenant:{user_id or 'anonymous'}:{base_key}"
        
        # Adicionar parâmetros adicionais se fornecidos
        if additional_params:
            # Ordenar para garantir consistência
            params_str = json.dumps(additional_params, sort_keys=True)
            # Hash para evitar chaves muito longas
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            cache_key += f":{params_hash}"
            
        return cache_key

    @classmethod
    def get_ttl(cls, cache_type: str) -> int:
        """Obter TTL baseado no tipo de cache"""
        return cls.TTL_CONFIG.get(cache_type, cls.TTL_CONFIG['default'])

    @classmethod  
    def set_cache(cls, key: str, value: Any, timeout: Optional[int] = None,
                  cache_type: str = 'default') -> bool:
        """
        Definir valor no cache com TTL inteligente
        
        Args:
            key: Chave do cache
            value: Valor a ser armazenado
            timeout: TTL customizado (usa TTL inteligente se None)
            cache_type: Tipo de cache para TTL automático
            
        Returns:
            True se bem-sucedido, False caso contrário
        """
        try:
            cache = cls.get_cache()
            if not cache:
                return False
                
            if timeout is None:
                timeout = cls.get_ttl(cache_type)
                
            cache.set(key, value, timeout=timeout)
            logger.debug(f"Cache SET: {key} (TTL: {timeout}s, tipo: {cache_type})")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao definir cache {key}: {str(e)}")
            return False

    @classmethod
    def get_cache_value(cls, key: str) -> Any:
        """Obter valor do cache"""
        try:
            cache = cls.get_cache()
            if not cache:
                return None
                
            value = cache.get(key)
            if value is not None:
                logger.debug(f"Cache HIT: {key}")
            else:
                logger.debug(f"Cache MISS: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Erro ao obter cache {key}: {str(e)}")
            return None

    @classmethod
    def delete_cache(cls, key: str) -> bool:
        """Deletar chave específica do cache"""
        try:
            cache = cls.get_cache()
            if not cache:
                return False
                
            cache.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao deletar cache {key}: {str(e)}")
            return False

    @classmethod
    def invalidate_tenant_cache(cls, user_id: int, pattern: Optional[str] = None) -> bool:
        """
        Invalidar cache de um tenant específico
        
        Args:
            user_id: ID do usuário/tenant
            pattern: Padrão opcional para invalidar apenas chaves específicas
            
        Returns:
            True se bem-sucedido
        """
        try:
            cache = cls.get_cache()
            if not cache:
                return False
                
            # Para SimpleCache, precisamos usar delete_many ou limpar manualmente
            # Para RedisCache, poderíamos usar pattern matching
            
            if hasattr(cache, 'clear'):
                # Não é ideal pois limpa todo o cache, mas funciona para desenvolvimento
                logger.warning(f"Limpando todo o cache (tenant {user_id}, pattern: {pattern})")
                cache.clear()
                return True
                
            logger.warning(f"Cache backend não suporta invalidação por padrão")
            return False
            
        except Exception as e:
            logger.error(f"Erro ao invalidar cache tenant {user_id}: {str(e)}")
            return False


def cached(cache_type: str = 'default', key_func: Optional[Callable] = None,
          timeout: Optional[int] = None):
    """
    Decorator para cache automático de funções
    
    Args:
        cache_type: Tipo de cache para TTL automático
        key_func: Função customizada para gerar chave (opcional)
        timeout: TTL customizado em segundos (opcional)
        
    Example:
        @cached('dashboard_stats')
        def get_dashboard_data(user_id):
            return expensive_calculation()
            
        @cached('silo_occupancy', key_func=lambda silo_id: f"silo:{silo_id}")
        def get_silo_occupancy(silo_id):
            return calculate_occupancy(silo_id)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Gerar chave de cache
            if key_func:
                cache_key_suffix = key_func(*args, **kwargs)
            else:
                # Chave padrão baseada no nome da função e argumentos
                func_name = func.__name__
                args_str = '_'.join(str(arg) for arg in args)
                kwargs_str = '_'.join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key_suffix = f"{func_name}:{args_str}:{kwargs_str}"
                
            # Criar chave com namespace de tenant
            cache_key = CacheManager.get_tenant_key(
                cache_key_suffix, 
                additional_params=kwargs if kwargs else None
            )
            
            # Tentar obter do cache
            cached_result = CacheManager.get_cache_value(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Cache miss - executar função original
            result = func(*args, **kwargs)
            
            # Armazenar no cache
            CacheManager.set_cache(
                cache_key, 
                result, 
                timeout=timeout,
                cache_type=cache_type
            )
            
            return result
            
        # Adicionar método para invalidar cache da função (corrigido para usar mesma lógica de chave)
        def invalidate_cache(*args, **kwargs):
            # Usar exatamente a mesma lógica de geração de chave do decorator
            if key_func:
                cache_key_suffix = key_func(*args, **kwargs)
            else:
                # Mesma lógica do decorator principal
                func_name = func.__name__
                args_str = '_'.join(str(arg) for arg in args)
                kwargs_str = '_'.join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key_suffix = f"{func_name}:{args_str}:{kwargs_str}"
                
            # Criar chave com namespace de tenant (mesma lógica)
            cache_key = CacheManager.get_tenant_key(
                cache_key_suffix, 
                additional_params=kwargs if kwargs else None
            )
            
            return CacheManager.delete_cache(cache_key)
            
        wrapper.invalidate_cache = invalidate_cache
        
        return wrapper
    return decorator


def cache_key_for_user(base_key: str, additional_params: Dict[str, Any] = None) -> str:
    """Helper para gerar chave de cache para o usuário atual"""
    return CacheManager.get_tenant_key(base_key, additional_params=additional_params)


def invalidate_user_cache(pattern: Optional[str] = None) -> bool:
    """Invalidar cache do usuário atual"""
    if not current_user.is_authenticated:
        return False
        
    return CacheManager.invalidate_tenant_cache(current_user.id, pattern)


# Event system para invalidação automática
class CacheInvalidationEvents:
    """Sistema de eventos para invalidação automática de cache"""
    
    @staticmethod
    def on_silo_movement(user_id: int, silo_id: int):
        """Invalidar cache relacionado a movimentações de silo"""
        # Invalidar funções específicas pelos nomes corretos
        cache_patterns = [
            # Dashboard statistics function
            f"get_dashboard_statistics:::",  # Função sem argumentos
            # Silo-specific cache
            f"silo_stock:{silo_id}",      # Cache específico do silo
            f"silo_percent:{silo_id}",    # Percentual do silo
            f"silo_capacity:{silo_id}",   # Capacidade do silo
        ]
        
        for pattern in cache_patterns:
            key = CacheManager.get_tenant_key(pattern, user_id)
            CacheManager.delete_cache(key)
            
        logger.info(f"Cache invalidado por movimentação silo {silo_id} (user {user_id})")

    @staticmethod  
    def on_supply_movement(user_id: int):
        """Invalidar cache relacionado a movimentações de insumos"""
        # Usar nomes corretos das funções
        cache_patterns = [
            f"get_dashboard_statistics:::",          # Dashboard statistics
            f"get_low_stock_supplies:::",           # Supply levels
            f"get_recent_supply_movements:::",      # Recent movements
        ]
        
        for pattern in cache_patterns:
            key = CacheManager.get_tenant_key(pattern, user_id)
            CacheManager.delete_cache(key)
            
        logger.info(f"Cache invalidado por movimentação insumo (user {user_id})")

    @staticmethod
    def on_user_data_change(user_id: int):
        """Invalidar cache relacionado a dados do usuário"""
        patterns = ['user_info', 'dashboard_stats']
        
        for pattern in patterns:
            key = CacheManager.get_tenant_key(pattern, user_id) 
            CacheManager.delete_cache(key)
            
        logger.info(f"Cache invalidado por mudança dados usuário {user_id}")