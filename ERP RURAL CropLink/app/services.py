"""
Módulo de Serviços - CropLink
Contém a lógica de negócio e serviços de dados do sistema agrícola.

Este módulo inclui:
- Serviços de dashboard e relatórios
- Agregação de dados
- Isolamento multi-tenant
- Funções de CRUD especializadas

Author: CropLink Development Team
Created: 2025-09-24
"""

import json
import logging
from datetime import datetime, timedelta
from flask_login import current_user
from flask import flash
from sqlalchemy import func, text
from .cache import cached

# Configuração de logging
logger = logging.getLogger(__name__)


# SISTEMA DE ISOLAMENTO DE DADOS POR USUÁRIO

def filtrar_por_usuario(query, modelo):
    """
    Aplica filtro de isolamento por usuário nas consultas.
    
    Implementa segurança multi-tenant garantindo que usuários vejam
    apenas seus próprios dados, exceto administradores.
    
    Args:
        query: Query do SQLAlchemy
        modelo: Modelo da tabela (ex: Insumo, Silo, etc.)
    
    Returns:
        Query: Query filtrada por user_id ou sem filtro para admins
    """
    # Administradores veem todos os dados (visão global)
    if current_user.is_authenticated and current_user.is_admin:
        return query
    
    # Usuários comuns veem apenas seus próprios dados
    if current_user.is_authenticated:
        return query.filter(modelo.user_id == current_user.id)
    
    # Caso não autenticado (não deveria acontecer com @login_required)
    return query.filter(False)  # Retorna vazio


def criar_com_usuario(model_cls, **kwargs):
    """
    Cria um novo registro atribuindo automaticamente o user_id do usuário logado.
    
    SEGURANÇA: Esta função previne ataques de privilege escalation forçando
    o user_id correto baseado no usuário logado.
    
    Args:
        model_cls: Classe do modelo (ex: Insumo, Silo, etc.)
        **kwargs: Campos do modelo
    
    Returns:
        object: Instância do modelo criada com user_id
        
    Raises:
        ValueError: Se usuário não estiver autenticado
    """
    if not current_user.is_authenticated:
        raise ValueError("Usuário deve estar autenticado para criar registros")
    
    # CORREÇÃO DE SEGURANÇA: Sempre forçar user_id do usuário atual para não-admins
    # Apenas administradores podem especificar user_id diferente do próprio
    if current_user.is_admin and 'target_user_id' in kwargs:
        # Administradores podem usar target_user_id para criar registros para outros usuários
        kwargs['user_id'] = kwargs.pop('target_user_id')
    else:
        # Para usuários comuns, sempre forçar o próprio user_id independente do que foi passado
        # Remove qualquer user_id passado maliciosamente
        if 'user_id' in kwargs:
            logger.warning(f"Tentativa de especificar user_id por usuário não-admin: {current_user.username}")
        kwargs['user_id'] = current_user.id
    
    return model_cls(**kwargs)


def obter_por_id_usuario(modelo, registro_id):
    """
    Obtém um registro por ID respeitando isolamento de usuário.
    
    Args:
        modelo: Classe do modelo
        registro_id: ID do registro
    
    Returns:
        object|None: Registro encontrado ou None se não pertencer ao usuário
    """
    query = modelo.query.filter(modelo.id == registro_id)
    return filtrar_por_usuario(query, modelo).first()


def contar_registros_usuario(modelo):
    """
    Conta total de registros do usuário para um modelo específico.
    
    Args:
        modelo: Classe do modelo
        
    Returns:
        int: Número total de registros
    """
    return filtrar_por_usuario(modelo.query, modelo).count()


def get_or_404_user_scoped(modelo, registro_id):
    """
    Obtém registro ou retorna 404 respeitando isolamento de usuário.
    
    Args:
        modelo: Classe do modelo
        registro_id: ID do registro
        
    Returns:
        object: Registro encontrado
        
    Raises:
        404: Se registro não encontrado ou não pertence ao usuário
    """
    from flask import abort
    record = obter_por_id_usuario(modelo, registro_id)
    if record is None:
        abort(404)
    return record


def validate_ownership(modelo, registro_id, field_name="registro"):
    """
    Valida se o registro pertence ao usuário atual.
    
    Args:
        modelo: Classe do modelo
        registro_id: ID do registro
        field_name: Nome do campo para mensagens de erro
        
    Returns:
        object: Registro validado
        
    Raises:
        ValueError: Se registro não encontrado ou não pertence ao usuário
    """
    record = obter_por_id_usuario(modelo, registro_id)
    
    if not record:
        logger.warning(f"Tentativa de acesso a {field_name} inexistente ou não autorizado. "
                      f"User: {current_user.username}, ID: {registro_id}")
        raise ValueError(f"{field_name.capitalize()} não encontrado ou acesso negado!")
    
    return record


def validate_parent_child_ownership(parent_model, parent_id, child_model, child_data):
    """
    Valida ownership entre registros pai e filho para prevenir ataques cross-user.
    
    Args:
        parent_model: Modelo do registro pai
        parent_id: ID do registro pai
        child_model: Modelo do registro filho
        child_data: Dados do registro filho
        
    Returns:
        object: Registro pai validado
        
    Raises:
        ValueError: Se validação falhar
    """
    # Validar ownership do registro pai
    parent_record = validate_ownership(parent_model, parent_id, parent_model.__name__.lower())
    
    # Verificar consistência de user_id entre pai e filho
    if child_data.get('user_id') != parent_record.user_id:
        logger.error(f"ERRO SEGURANÇA: user_id inconsistente entre pai ({parent_record.user_id}) "
                    f"e filho ({child_data.get('user_id')})")
        raise ValueError("Erro de consistência de dados!")
    
    return parent_record


# SERVIÇOS DE DASHBOARD

@cached('dashboard_stats')
def get_dashboard_statistics():
    """
    Obtém as estatísticas principais do dashboard (contadores de entidades).
    
    Coleta contadores de todas as entidades principais do sistema agrícola
    para exibição no dashboard principal.
    
    CACHE: TTL 60s - dados dinâmicos atualizados frequentemente
    
    Returns:
        tuple: (total_supplies, total_machines, total_employees, total_silos, low_stock_count)
               - total_supplies: Número total de insumos
               - total_machines: Número total de máquinas  
               - total_employees: Número total de funcionários
               - total_silos: Número total de silos
               - low_stock_count: Número de insumos com estoque baixo
    """
    from .models import Insumo, Maquinario, Funcionario, Silo
    
    total_insumos = contar_registros_usuario(Insumo)
    total_maquinas = contar_registros_usuario(Maquinario)
    total_funcionarios = contar_registros_usuario(Funcionario)
    total_silos = contar_registros_usuario(Silo)
    insumos_baixo_estoque_count = filtrar_por_usuario(
        Insumo.query.filter(Insumo.quantidade < 10), Insumo
    ).count()
    
    return (total_insumos, total_maquinas, total_funcionarios, total_silos, insumos_baixo_estoque_count)


@cached('supply_levels')  
def get_low_stock_supplies():
    """
    Obtém lista de insumos com baixo estoque (< 10 unidades).
    
    Identifica insumos que precisam de reposição para alertas no dashboard.
    
    CACHE: TTL 120s - mudanças moderadas nos níveis de estoque
    
    Returns:
        list: Lista de objetos Insumo com quantidade < 10 unidades,
              limitada aos primeiros 20 registros para performance
    """
    from .models import Insumo
    
    return filtrar_por_usuario(
        Insumo.query.filter(Insumo.quantidade < 10), Insumo
    ).limit(20).all()


@cached('supply_levels')
def get_recent_supply_movements():
    """
    Obtém as últimas movimentações de insumos do usuário.
    
    Busca as 10 movimentações mais recentes com dados do insumo relacionado
    para exibição no feed de atividades do dashboard.
    
    CACHE: TTL 120s - mudanças moderadas nas movimentações
    
    Returns:
        list: Lista de objetos MovimentacaoInsumo ordenados por data decrescente,
              limitada a 10 registros com dados de insumo carregados
    """
    from .models import MovimentacaoInsumo
    from app import db
    from sqlalchemy.orm import joinedload
    
    return filtrar_por_usuario(
        db.session.query(MovimentacaoInsumo)
        .options(joinedload(MovimentacaoInsumo.insumo))
        .order_by(MovimentacaoInsumo.data.desc()),
        MovimentacaoInsumo
    ).limit(10).all()


def obter_agregacao_movimentacoes(data_inicio):
    """
    Obtém dados agregados de movimentações para o período especificado.
    
    Args:
        data_inicio (datetime): Data inicial para busca
        
    Returns:
        list: Lista de tuplas (data_mov, tipo, total_quantidade)
    """
    if current_user.is_admin:
        mov_stats_query = text("""
            SELECT 
                DATE(data) as data_mov,
                tipo,
                SUM(quantidade) as total_quantidade
            FROM movimentacao_insumo 
            WHERE data >= :data_inicio
            GROUP BY DATE(data), tipo
            ORDER BY data_mov DESC
        """)
        from app import db
        return db.session.execute(mov_stats_query, {'data_inicio': data_inicio}).fetchall()
    else:
        mov_stats_query = text("""
            SELECT 
                DATE(data) as data_mov,
                tipo,
                SUM(quantidade) as total_quantidade
            FROM movimentacao_insumo 
            WHERE data >= :data_inicio AND user_id = :user_id
            GROUP BY DATE(data), tipo
            ORDER BY data_mov DESC
        """)
        from app import db
        return db.session.execute(mov_stats_query, {
            'data_inicio': data_inicio,
            'user_id': current_user.id
        }).fetchall()


def processar_dados_movimentacoes_grafico(movimentacoes_agregadas):
    """
    Processa dados de movimentações para exibição em gráfico.
    
    Args:
        movimentacoes_agregadas (list): Lista de tuplas de movimentações agregadas
        
    Returns:
        tuple: (labels_movimentacoes, dados_entradas, dados_saidas)
    """
    # Inicializar dados para os últimos 7 dias
    dados_movimentacoes = {}
    for i in range(7):
        data_atual = (datetime.utcnow() - timedelta(days=6 - i)).date()
        dados_movimentacoes[data_atual] = {'entradas': 0, 'saidas': 0}

    # Preencher com dados reais
    for row in movimentacoes_agregadas:
        data_mov = row[0].date() if hasattr(row[0], 'date') else row[0]
        tipo = row[1]
        quantidade = float(row[2])

        if data_mov in dados_movimentacoes:
            if tipo == 'Entrada':
                dados_movimentacoes[data_mov]['entradas'] = quantidade
            else:  # Saída
                dados_movimentacoes[data_mov]['saidas'] = quantidade

    # Preparar dados para gráfico
    labels_movimentacoes = []
    dados_entradas = []
    dados_saidas = []

    for data in sorted(dados_movimentacoes.keys()):
        labels_movimentacoes.append(data.strftime('%d/%m'))
        dados_entradas.append(dados_movimentacoes[data]['entradas'])
        dados_saidas.append(dados_movimentacoes[data]['saidas'])

    return (labels_movimentacoes, dados_entradas, dados_saidas)


def obter_agregacao_chuva(data_inicio):
    """
    Obtém dados agregados de chuva para o período especificado.
    
    Args:
        data_inicio (datetime): Data inicial para busca
        
    Returns:
        list: Lista de tuplas (data_chuva, total_mm)
    """
    if current_user.is_admin:
        chuva_stats_query = text("""
            SELECT 
                DATE(data) as data_chuva,
                SUM(quantidade_mm) as total_mm
            FROM registro_chuva 
            WHERE data >= :data_inicio
            GROUP BY DATE(data)
            ORDER BY data_chuva DESC
        """)
        from app import db
        return db.session.execute(chuva_stats_query, {'data_inicio': data_inicio}).fetchall()
    else:
        chuva_stats_query = text("""
            SELECT 
                DATE(data) as data_chuva,
                SUM(quantidade_mm) as total_mm
            FROM registro_chuva 
            WHERE data >= :data_inicio AND user_id = :user_id
            GROUP BY DATE(data)
            ORDER BY data_chuva DESC
        """)
        from app import db
        return db.session.execute(chuva_stats_query, {
            'data_inicio': data_inicio,
            'user_id': current_user.id
        }).fetchall()


def processar_dados_chuva_grafico(chuva_agregada):
    """
    Processa dados de chuva para exibição em gráfico e estatísticas.
    
    Args:
        chuva_agregada (list): Lista de tuplas de chuva agregada
        
    Returns:
        tuple: (total_chuva_semana, media_chuva_semana, labels_chuva, dados_chuva)
    """
    # Calcular estatísticas
    total_chuva_semana = sum(row[1] for row in chuva_agregada)
    media_chuva_semana = round(total_chuva_semana / 7, 1) if chuva_agregada else 0

    # Criar dicionário para acesso rápido
    dados_chuva_dict = {
        row[0].date() if hasattr(row[0], 'date') else row[0]: float(row[1])
        for row in chuva_agregada
    }

    # Preparar dados para gráfico
    labels_chuva = []
    dados_chuva = []

    for i in range(7):
        data_atual = (datetime.utcnow() - timedelta(days=6 - i)).date()
        labels_chuva.append(data_atual.strftime('%d/%m'))
        dados_chuva.append(dados_chuva_dict.get(data_atual, 0))

    return (total_chuva_semana, media_chuva_semana, labels_chuva, dados_chuva)


def obter_registros_chuva_recentes(data_inicio):
    """
    Obtém os registros de chuva recentes do usuário.
    
    Args:
        data_inicio (datetime): Data inicial para busca
        
    Returns:
        list: Lista de objetos RegistroChuva ordenados por data decrescente
    """
    from .models import RegistroChuva
    
    return filtrar_por_usuario(
        RegistroChuva.query.filter(RegistroChuva.data >= data_inicio),
        RegistroChuva
    ).order_by(RegistroChuva.data.desc()).limit(10).all()