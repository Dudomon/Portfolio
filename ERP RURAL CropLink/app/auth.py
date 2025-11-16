"""
Módulo de Autenticação - CropLink
Contém decoradores, validadores e lógica de autorização do sistema.

Este módulo inclui:
- Decoradores de autorização
- Configuração de planos
- Validadores de acesso
- Funções de autenticação

Author: CropLink Development Team
Created: 2025-09-24
"""

import logging
from functools import wraps
from flask import redirect, url_for, flash, request
from flask_login import current_user, logout_user

# Configuração de logging
logger = logging.getLogger(__name__)

# SISTEMA DE PLANOS POR ASSINATURA
PLANOS_CONFIG = {
    'basic': {
        'nome': 'Plano Básico',
        'modulos': ['silos', 'graos', 'dashboard'],
        'preco': 'Gratuito',
        'limite_silos': 5,
        'descricao': 'Gestão de Silos e Movimentação de Grãos'
    },
    'plus': {
        'nome': 'Plano Plus', 
        'modulos': ['silos', 'graos', 'pulverizacao', 'dashboard'],
        'preco': 'Em breve',
        'limite_silos': 20,
        'descricao': 'Básico + Módulo de Pulverização',
        'status': 'em_breve'
    },
    'premium': {
        'nome': 'Plano Premium',
        'modulos': ['silos', 'graos', 'pulverizacao', 'caderno_campo', 'bolsa_valores', 'dashboard'],
        'preco': 'Em breve', 
        'limite_silos': 'ilimitado',
        'descricao': 'Acesso Completo + Caderno de Campo + Financeiro Integrado',
        'status': 'em_breve'
    }
}


def requer_plano(modulos_permitidos):
    """
    Decorator para verificar se o usuário tem acesso ao módulo baseado no plano.
    
    SEGURANÇA: Este decorator DEVE incluir verificação de autenticação antes
    de verificar planos, pois precisa acessar propriedades do current_user.
    
    Args:
        modulos_permitidos (list): Lista de módulos que têm acesso à rota
    
    Usage:
        @requer_plano(['plus', 'premium'])
        def rota_premium():
            pass
    """
    def decorator(f):
        @wraps(f)
        @login_required  # CRÍTICO: Autenticação obrigatória antes de verificar planos
        def decorated_function(*args, **kwargs):
            # Verificar se a assinatura está válida
            if not current_user.esta_assinatura_valida():
                flash('Sua assinatura expirou. Renove para continuar usando o sistema.', 'danger')
                return redirect(url_for('upgrade_plano'))
            
            # Verificar se o plano atual permite acesso
            tem_acesso = False
            for modulo in modulos_permitidos:
                if current_user.tem_acesso_modulo(modulo):
                    tem_acesso = True
                    break
            
            if not tem_acesso:
                plano_atual = PLANOS_CONFIG.get(current_user.plano, {})
                flash(f'Esta funcionalidade não está disponível no {plano_atual.get("nome", "seu plano atual")}. Faça upgrade para acessar!', 'warning')
                return redirect(url_for('upgrade_plano'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def admin_required(f):
    """
    Decorator para verificar se o usuário tem privilégios administrativos.
    
    Args:
        f: Função a ser decorada
        
    Returns:
        function: Função decorada com validação de admin
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        if not current_user.esta_aprovado():
            logout_user()
            flash('Acesso negado.', 'danger')
            return redirect(url_for('login'))
            
        if not current_user.is_admin:
            flash('Acesso negado. Privilégios administrativos necessários.', 'danger')
            return redirect(url_for('dashboard'))
            
        return f(*args, **kwargs)
    return decorated_function


def super_admin_required(f):
    """
    Decorator para rotas que requerem privilégios de Super Administrador.
    
    Args:
        f: Função a ser decorada
        
    Returns:
        function: Função decorada com validação de super admin
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        if not current_user.esta_aprovado():
            logout_user()
            flash('Acesso negado.', 'danger')
            return redirect(url_for('login'))
            
        if not current_user.is_super_admin():
            flash('Acesso negado. Apenas Super Administradores podem acessar esta funcionalidade.', 'danger')
            return redirect(url_for('dashboard'))
            
        return f(*args, **kwargs)
    return decorated_function


def aprovacao_required(f):
    """
    Decorator para verificar se usuário está aprovado.
    
    Args:
        f: Função a ser decorada
        
    Returns:
        function: Função decorada com validação de aprovação
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        if not current_user.esta_aprovado():
            if current_user.status_aprovacao == 'pendente':
                flash('Seu cadastro está aguardando aprovação do administrador.', 'warning')
            elif current_user.status_aprovacao == 'rejeitado':
                flash('Seu cadastro foi rejeitado. Entre em contato com o administrador.', 'danger')
            else:
                flash('Seu acesso não está ativo. Entre em contato com o administrador.', 'danger')
            logout_user()  # Força logout de usuários não aprovados
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function


def requer_nivel_plano(nivel_minimo):
    """
    Decorator para verificar nível mínimo do plano.
    
    Args:
        nivel_minimo (str): Nível mínimo requerido ('basic', 'plus', 'premium')
        
    Returns:
        function: Decorator para validação de nível
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            
            # Verificar se a assinatura está válida
            if not current_user.esta_assinatura_valida():
                flash('Sua assinatura expirou. Renove para continuar usando o sistema.', 'danger')
                return redirect(url_for('upgrade_plano'))
            
            # Mapeamento de níveis
            niveis = {'basic': 1, 'plus': 2, 'premium': 3}
            nivel_usuario = niveis.get(current_user.plano, 0)
            nivel_necessario = niveis.get(nivel_minimo, 999)
            
            if nivel_usuario < nivel_necessario:
                flash(f'Esta funcionalidade requer pelo menos o plano {nivel_minimo.title()}. Faça upgrade!', 'warning')
                return redirect(url_for('upgrade_plano'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def verificar_acesso_obrigatorio():
    """
    Middleware para garantir verificação de email e troca de senha obrigatória.

    Esta função deve ser registrada como before_request no Flask.
    """
    try:
        # Ignorar rotas estáticas e de autenticação
        endpoints_publicos = [
            'static', 'health_check', 'login', 'register', 'logout',
            'verificar_email', 'trocar_senha_primeiro_acesso'
        ]

        if request.endpoint in endpoints_publicos:
            return

        # Se usuário está logado
        if current_user.is_authenticated:
            from flask import current_app
            current_app.logger.info(f"🔍 Verificando acesso para usuário: {current_user.username}")

            # Se o email não foi verificado, redirecionar para logout (segurança)
            try:
                if not current_user.email_verificado:
                    current_app.logger.warning(f"❌ Email não verificado: {current_user.username}")
                    logout_user()
                    flash('Sua conta precisa de verificação de e-mail. Entre em contato com o administrador.', 'warning')
                    return redirect(url_for('login'))
            except Exception as e:
                current_app.logger.error(f"❌ ERRO verificando email_verificado: {str(e)}")
                raise

            # Se não pode fazer login (não aprovado ou inativo), fazer logout
            try:
                pode_logar = current_user.pode_fazer_login()
                current_app.logger.info(f"🔑 pode_fazer_login() = {pode_logar}")

                if not pode_logar:
                    current_app.logger.warning(f"❌ Usuário não pode fazer login: {current_user.username}")
                    logout_user()
                    flash('Sua conta não está ativa ou não foi aprovada. Entre em contato com o administrador.', 'warning')
                    return redirect(url_for('login'))
            except Exception as e:
                current_app.logger.error(f"❌ ERRO CRÍTICO em pode_fazer_login(): {str(e)}")
                current_app.logger.error(f"   Tipo de erro: {type(e).__name__}")
                current_app.logger.error(f"   Usuário: {current_user.username}")
                import traceback
                current_app.logger.error(f"   Traceback: {traceback.format_exc()}")
                raise

            # Se precisa trocar senha no primeiro acesso
            try:
                if current_user.precisa_trocar_senha() and request.endpoint != 'trocar_senha_primeiro_acesso':
                    return redirect(url_for('trocar_senha_primeiro_acesso'))
            except Exception as e:
                current_app.logger.error(f"❌ ERRO em precisa_trocar_senha(): {str(e)}")
                raise

    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"🔥 ERRO FATAL em verificar_acesso_obrigatorio: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        current_app.logger.error(traceback_str)

        # TEMPORÁRIO: Mostrar erro na tela para debug em produção
        logout_user()
        flash(f'ERRO DE SISTEMA (temporário para debug): {type(e).__name__}: {str(e)[:200]}', 'danger')
        flash(f'Por favor, tire um print desta tela e envie para o desenvolvedor', 'warning')
        return redirect(url_for('login'))