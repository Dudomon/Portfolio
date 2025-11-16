"""
Módulo de Utilidades - CropLink
Contém funções auxiliares reutilizáveis para o sistema de gestão agrícola.

Este módulo inclui:
- Validação de ambiente de produção
- Helpers de segurança e validação
- Filtros de template
- Utilitários gerais

Author: CropLink Development Team
Created: 2025-09-24
"""

import os
import logging
from werkzeug.utils import secure_filename

# Configuração de logging
logger = logging.getLogger(__name__)

# Extensões de arquivo permitidas para upload
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'xlsx', 'xls', 'csv'}


def validate_production_environment():
    """
    Valida se todas as variáveis de ambiente críticas estão configuradas.
    
    Esta função verifica variáveis essenciais para produção e emite avisos
    ou falha dependendo do ambiente (desenvolvimento vs produção).
    
    Raises:
        RuntimeError: Em produção, se variáveis obrigatórias estão faltando
    """
    required_vars = [
        'SECRET_KEY',
        'DATABASE_URL', 
        'ROOT_ADMIN_PASSWORD',
        'ALOIZIO_ADMIN_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or (var == 'SECRET_KEY' and value == 'dev-secret-key-change-in-production'):
            missing_vars.append(var)
    
    if missing_vars:
        error_msg = f"ERRO DE SEGURANÇA: Variáveis obrigatórias não configuradas: {', '.join(missing_vars)}"
        logger.error(error_msg)
        logger.error("Configure essas variáveis no Secrets do Replit antes do deploy!")
        
        # Em produção, falha imediatamente. Em desenvolvimento, apenas avisa.
        if os.getenv('FLASK_ENV') == 'production' or os.getenv('REPL_DEPLOYMENT') == '1':
            raise RuntimeError(error_msg)
        else:
            logger.warning("AVISO: Executando em modo desenvolvimento com configurações inseguras")


def allowed_file(filename):
    """
    Verifica se o arquivo tem uma extensão permitida.
    
    Args:
        filename (str): Nome do arquivo a ser verificado
        
    Returns:
        bool: True se a extensão for permitida, False caso contrário
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def secure_filename_custom(filename):
    """
    Cria um nome de arquivo seguro com tratamento personalizado.
    
    Args:
        filename (str): Nome original do arquivo
        
    Returns:
        str: Nome de arquivo seguro
    """
    # Remove caracteres perigosos e normaliza
    filename = secure_filename(filename)
    
    # Adiciona timestamp se necessário para evitar conflitos
    if not filename:
        filename = "arquivo_sem_nome"
    
    return filename


def formatar_numero_extenso(n):
    """
    Formata números para exibição em formato extenso brasileiro.
    
    Converte números para formato legível:
    - < 1000: mantém formato original
    - >= 1000: mostra em milhares (ex: 1,5 mil)
    - >= 1000000: mostra em milhões (ex: 2,3 milhões)
    
    Args:
        n (int|float): Número a ser formatado
        
    Returns:
        str: Número formatado em string
    """
    if not isinstance(n, (int, float)):
        return n
        
    if abs(n) < 1000:
        return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".").replace(",00", "")
    elif abs(n) < 1000000:
        return f"{n/1000:,.1f}".replace(".", ",").replace(",0", "") + " mil"
    else:
        return f"{n/1000000:,.1f}".replace(".", ",").replace(",0", "") + " milhões"


def add_security_and_cache_headers(response):
    """
    Adiciona cabeçalhos de segurança e cache às respostas HTTP.
    
    Args:
        response: Objeto de resposta Flask
        
    Returns:
        Response: Resposta com cabeçalhos de segurança adicionados
    """
    from flask import request
    
    # Cabeçalhos de segurança
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Cache control para arquivos estáticos
    if request.endpoint == 'static':
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        response.headers['X-Frame-Options'] = 'DENY'  # Para arquivos estáticos
    else:
        # Para outras rotas, evitar cache mas permitir CSRF tokens
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        # Headers específicos para CSRF e iframe
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Permite iframes do mesmo domínio
        response.headers['Cross-Origin-Embedder-Policy'] = 'unsafe-none'
    
    return response


def health_check_database(db):
    """
    Verifica a conectividade com o banco de dados.
    
    Args:
        db: Instância do SQLAlchemy
        
    Returns:
        tuple: (status_bool, message)
    """
    try:
        from sqlalchemy import text
        db.session.execute(text('SELECT 1'))
        db.session.commit()
        return True, "database connected"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False, f"database connection failed: {str(e)}"


def validate_email_format(email):
    """
    Valida formato básico de email.
    
    Args:
        email (str): Email a ser validado
        
    Returns:
        bool: True se o formato estiver válido
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password_strength(password):
    """
    Valida força da senha conforme critérios de segurança.
    
    Args:
        password (str): Senha a ser validada
        
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    if not password or len(password) < 8:
        errors.append('A senha deve ter pelo menos 8 caracteres.')
        
    if password and not any(c.isupper() for c in password):
        errors.append('A senha deve conter pelo menos uma letra maiúscula.')
        
    if password and not any(c.islower() for c in password):
        errors.append('A senha deve conter pelo menos uma letra minúscula.')
        
    if password and not any(c.isdigit() for c in password):
        errors.append('A senha deve conter pelo menos um número.')
        
    if password and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
        errors.append('A senha deve conter pelo menos um caractere especial.')
    
    return len(errors) == 0, errors