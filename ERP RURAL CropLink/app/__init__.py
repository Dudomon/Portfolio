#!/usr/bin/env python3
"""
Sistema de Gestão Agrícola CropLink - Aplicação Principal
Versão modular com funcionalidades organizadas em módulos separados.

Este arquivo contém a configuração principal da aplicação Flask e as rotas,
utilizando módulos auxiliares para utils, services e auth.
"""

import os
import logging
import urllib.parse
import json
import csv
import io
import locale
from datetime import datetime, timedelta

# Configurar locale para português brasileiro
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_TIME, 'pt_BR')
        except locale.Error:
            pass  # Se não conseguir, mantém o locale padrão
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect
from flask_mail import Mail, Message
from flask_caching import Cache
from sqlalchemy import Date, Time, func, text
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv
from functools import wraps

from flask_cors import CORS  # Importa a extensão CORS

# Importar módulos auxiliares
from .utils import validate_production_environment, formatar_numero_extenso, health_check_database
from .auth import (
    PLANOS_CONFIG, verificar_acesso_obrigatorio, admin_required, 
    super_admin_required, aprovacao_required, requer_plano, requer_nivel_plano
)
from .services import (
    get_dashboard_statistics, get_low_stock_supplies,
    get_recent_supply_movements, obter_agregacao_movimentacoes,
    processar_dados_movimentacoes_grafico, obter_agregacao_chuva,
    processar_dados_chuva_grafico, obter_registros_chuva_recentes,
    filtrar_por_usuario, criar_com_usuario, obter_por_id_usuario,
    contar_registros_usuario, get_or_404_user_scoped, validate_ownership,
    validate_parent_child_ownership
)
from .cache import CacheManager, cached, CacheInvalidationEvents

# Carrega variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validação de ambiente já importada do módulo utils

# Executar validação
validate_production_environment()


class Base(DeclarativeBase):
    pass


# Criar aplicação Flask
app = Flask(__name__, template_folder='../templates', static_folder='../static')
logger.info("Iniciando aplicação CropLink...")

# Configuração CORS adaptativa - detecta automaticamente o ambiente
dev_domain = os.getenv('REPLIT_DEV_DOMAIN', '')  # Domínio completo do preview/dev
replit_env = os.getenv('REPLIT_ENVIRONMENT', '')
is_deployment = os.getenv('REPL_DEPLOYMENT') == '1'
is_preview = dev_domain and not is_deployment

# Configurar origins baseado no ambiente detectado
if is_deployment:
    # Produção real - domínios específicos
    allowed_origins = [
        'https://fazenda-rebelato-production.replit.app',
        'https://seu-dominio-personalizado.com'
    ]
    cors_mode = "produção"
elif is_preview:
    # Preview environment - domínio do preview + workspace
    allowed_origins = [
        f"https://{dev_domain}",  # Domínio exato do preview
        'https://workspace.replit.dev',  # Workspace iframe
        f"https://{dev_domain.split('-')[0]}.replit.dev"  # Variações
    ]
    cors_mode = "preview"
else:
    # Desenvolvimento local
    allowed_origins = [
        'http://localhost:*', 
        'http://127.0.0.1:*',
        'https://workspace.replit.dev'
    ]
    cors_mode = "desenvolvimento"

CORS(app, 
     origins=allowed_origins, 
     supports_credentials=True,
     allow_headers=['Content-Type', 'X-CSRFToken', 'Authorization', 'X-Requested-With'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     expose_headers=['X-CSRFToken'])

logger.info(f"CORS configurado para {cors_mode}")
logger.info(f"Domínios permitidos: {allowed_origins}")
logger.info(f"Ambiente detectado - REPLIT_ENV: {replit_env}, DEV_DOMAIN: {dev_domain}, DEPLOYMENT: {is_deployment}")

# Configuração híbrida do banco de dados
database_url = os.getenv('DATABASE_URL')
if database_url:
    logger.info("Usando DATABASE_URL para conexão com banco")
    # Converter postgres:// para postgresql:// se necessário
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        logger.info("Convertido postgres:// para postgresql://")
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    logger.info("Usando configuração local do banco de dados")
    # Configuração local usando variáveis individuais
    db_user = os.getenv('DB_USER', 'postgres')
    db_pass_raw = os.getenv('DB_PASS', 'password')
    db_pass = urllib.parse.quote_plus(db_pass_raw) if db_pass_raw else ''
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'fazenda_db')
    app.config[
        'SQLALCHEMY_DATABASE_URI'] = f'postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY',
                                     'dev-secret-key-change-in-production')

# Configuração de email
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', '587'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'False').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', 'noreply@croplink.com')

app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_size': 10,  # Otimização: Pool maior
    'max_overflow': 20,  # Permite mais conexões quando necessário
}

# Inicializar extensões
db = SQLAlchemy(app, model_class=Base)
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)
mail = Mail(app)

# Configuração CSRF para melhor compatibilidade
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # 1 hora (mais tempo para evitar expiração)
app.config['WTF_CSRF_SSL_STRICT'] = False  # Permite HTTPS e HTTP
app.config['WTF_CSRF_CHECK_DEFAULT'] = True

# Configuração de sessão otimizada para preview/iframe
app.config['SESSION_COOKIE_SECURE'] = True  # Apenas HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Segurança XSS
app.config['SESSION_COOKIE_SAMESITE'] = 'None'  # Permite iframe cross-origin
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # 2 horas de sessão

# Headers adicionais para preview environment
if is_preview:
    app.config['SESSION_COOKIE_DOMAIN'] = None  # Permite qualquer domínio
    logger.info("Configuração de sessão otimizada para preview environment")

# Configuração do sistema de cache hierárquico L2 (Redis/SimpleCache)
cache_config = {
    'CACHE_TYPE': 'SimpleCache',  # Fallback para desenvolvimento
    'CACHE_DEFAULT_TIMEOUT': 300,  # 5 minutos padrão
    'CACHE_KEY_PREFIX': 'croplink:',
    'CACHE_REDIS_URL': os.getenv('REDIS_URL', None)
}

# Se Redis disponível, usar Redis como backend
if cache_config['CACHE_REDIS_URL']:
    cache_config['CACHE_TYPE'] = 'RedisCache'
    logger.info("Sistema de cache configurado com Redis")
else:
    logger.info("Sistema de cache configurado com SimpleCache (desenvolvimento)")

cache = Cache(app, config=cache_config)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
login_manager.login_message = "Por favor, faça o login para aceder a esta página."


# Registrar middleware importado do módulo auth
app.before_request(verificar_acesso_obrigatorio)


# Importar e inicializar modelos
from .models import init_models

models = init_models(db)

# Disponibilizar os modelos globalmente
Usuario = models['Usuario']
ProdutorRural = models['ProdutorRural']  # Sistema hierárquico
Insumo = models['Insumo']
MovimentacaoInsumo = models['MovimentacaoInsumo']
InsumoAgricola = models['InsumoAgricola']
MovimentacaoInsumoAgricola = models['MovimentacaoInsumoAgricola']
AplicacaoInsumo = models['AplicacaoInsumo']  # Modelo de aplicação de insumos
Maquinario = models['Maquinario']
Funcionario = models['Funcionario']
Diarista = models['Diarista']
RegistroDiaria = models['RegistroDiaria']
Silo = models['Silo']
Grao = models['Grao']
MovimentacaoSilo = models['MovimentacaoSilo']
RegistroChuva = models['RegistroChuva']
Talhao = models['Talhao']
Cliente = models['Cliente']
TransacaoCliente = models['TransacaoCliente']
ContasPagar = models['ContasPagar']
ContasReceber = models['ContasReceber']
Fornecedor = models['Fornecedor']


# Função para inicializar banco de dados e dados padrão
def initialize_database():
    """
    Inicializa o banco de dados e cria usuários administrativos padrão.
    
    Esta função é crítica para o setup inicial do sistema, realizando:
    1. Criação de todas as tabelas do banco de dados
    2. Setup de usuários administrativos usando variáveis de ambiente
    3. Garantia de privilégios administrativos para usuários existentes
    
    Usuários administrativos criados:
    - Root: Administrador principal (ROOT_ADMIN_PASSWORD)
    - Aloiziotadeu: Segundo administrador (ALOIZIO_ADMIN_PASSWORD)
    
    Segurança:
    - Senhas obrigatórias via variáveis de ambiente
    - Nunca expõe senhas em logs
    - Verifica se usuários já existem antes de criar
    
    Raises:
        Exception: Se houver falha na criação das tabelas ou usuários
        
    Note:
        Deve ser executada dentro do contexto da aplicação Flask
        Se as variáveis de ambiente não estiverem definidas, os admins não são criados
    """
    try:
        with app.app_context():
            # Criar tabelas
            db.create_all()
            logger.info("Tabelas criadas/verificadas com sucesso")

            # SEGURANÇA: Criação de administradores usando variáveis de ambiente
            admin_users_created = False
            
            # Root admin - usar senha da variável de ambiente
            root_password = os.getenv('ROOT_ADMIN_PASSWORD')
            if root_password:
                root = Usuario.query.filter_by(username='Root').first()
                if not root:
                    root = Usuario(username='Root',
                                   password_hash=bcrypt.generate_password_hash(root_password).decode('utf-8'),
                                   nome_completo='Administrador Root',
                                   email='root@fazendarebelato.com.br',
                                   status_aprovacao='aprovado',
                                   is_admin=True)
                    db.session.add(root)
                    logger.info("Usuário Root criado com privilégios administrativos")
                    admin_users_created = True
                else:
                    # Garantir que usuários existentes tenham privilégios de admin
                    if not root.is_admin:
                        root.is_admin = True
                        logger.info("Privilégios administrativos concedidos ao usuário Root")
            else:
                logger.info("ROOT_ADMIN_PASSWORD não definida - administrador Root não criado")

            # Segundo admin - usar senha da variável de ambiente
            aloizio_password = os.getenv('ALOIZIO_ADMIN_PASSWORD')
            if aloizio_password:
                aloizio = Usuario.query.filter_by(username='Aloiziotadeu').first()
                if not aloizio:
                    aloizio = Usuario(username='Aloiziotadeu',
                                      password_hash=bcrypt.generate_password_hash(aloizio_password).decode('utf-8'),
                                      nome_completo='Aloizio Tadeu',
                                      email='aloizio@fazenda.com',
                                      status_aprovacao='aprovado',
                                      is_admin=True)
                    db.session.add(aloizio)
                    logger.info("Usuário Aloiziotadeu criado com privilégios administrativos")
                    admin_users_created = True
                else:
                    # Garantir que usuários existentes tenham privilégios de admin
                    if not aloizio.is_admin:
                        aloizio.is_admin = True
                        logger.info("Privilégios administrativos concedidos ao usuário Aloiziotadeu")
            else:
                logger.info("ALOIZIO_ADMIN_PASSWORD não definida - administrador Aloiziotadeu não criado")
            
            if not admin_users_created:
                logger.warning("AVISO SEGURANÇA: Nenhum administrador foi criado. Defina ROOT_ADMIN_PASSWORD e/ou ALOIZIO_ADMIN_PASSWORD nas variáveis de ambiente para criar contas administrativas.")

            try:
                db.session.commit()
                logger.info("Usuários padrão verificados")
            except Exception as commit_error:
                db.session.rollback()
                logger.warning(f"Erro ao criar usuários padrão (provavelmente já existem): {commit_error}")

    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}")
        db.session.rollback()


# Inicializar banco de dados
initialize_database()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Usuario, int(user_id))


# Configuração dos planos importada do módulo auth

# Decoradores e funções de segurança importados dos módulos auth e services

def obter_por_id_usuario(modelo, registro_id):
    """
    Obtém um registro por ID respeitando isolamento de usuário
    
    Args:
        modelo: Classe do modelo
        registro_id: ID do registro
    
    Returns:
        Registro encontrado ou None se não pertencer ao usuário
    """
    query = modelo.query.filter(modelo.id == registro_id)
    return filtrar_por_usuario(query, modelo).first()

def contar_registros_usuario(modelo):
    """
    Conta registros do usuário atual
    
    Args:
        modelo: Classe do modelo
    
    Returns:
        Número de registros do usuário (ou todos para admin)
    """
    query = modelo.query
    return filtrar_por_usuario(query, modelo).count()

def get_or_404_user_scoped(modelo, registro_id):
    """
    Obtém um registro respeitando isolamento de usuário ou retorna 404
    
    SEGURANÇA: Esta função garante que usuários só possam acessar seus próprios registros.
    Administradores podem acessar qualquer registro.
    
    Args:
        modelo: Classe do modelo (ex: Insumo, Silo, etc.)
        registro_id: ID do registro
    
    Returns:
        Registro encontrado
    
    Raises:
        404: Se registro não existir ou não pertencer ao usuário
    """
    from flask import abort
    
    if not current_user.is_authenticated:
        abort(401)  # Unauthorized
    
    record = obter_por_id_usuario(modelo, registro_id)
    if not record:
        logger.warning(f"Tentativa de acesso não autorizado: usuário {current_user.username} tentou acessar {modelo.__name__} ID {registro_id}")
        abort(404)  # Not found - não revelar se existe ou não por segurança
    
    return record

def validate_ownership(modelo, registro_id, field_name="registro"):
    """
    Valida que um registro pertence ao usuário atual
    
    Args:
        modelo: Classe do modelo
        registro_id: ID do registro
        field_name: Nome do campo para mensagens de erro
    
    Returns:
        Registro se válido
    
    Raises:
        ValueError: Se registro não pertencer ao usuário
    """
    record = obter_por_id_usuario(modelo, registro_id)
    if not record:
        raise ValueError(f"{field_name.capitalize()} não encontrado ou acesso negado!")
    return record

def validate_parent_child_ownership(parent_model, parent_id, child_model, child_data):
    """
    Valida que registros pai e filho pertencem ao mesmo usuário
    
    SEGURANÇA: Previne ataques onde um usuário tenta criar registros filhos
    que referenciem registros pais de outros usuários.
    
    Args:
        parent_model: Classe do modelo pai (ex: Insumo)
        parent_id: ID do registro pai
        child_model: Classe do modelo filho (ex: MovimentacaoInsumo)  
        child_data: Dicionário com dados do registro filho
    
    Returns:
        parent_record: O registro pai validado
    
    Raises:
        ValueError: Se validação falhar
    """
    # Verificar se registro pai existe e pertence ao usuário
    parent_record = validate_ownership(parent_model, parent_id, parent_model.__name__)
    
    # Para não-admins, verificar que user_id no child_data corresponde ao usuário atual
    if not current_user.is_admin:
        if 'user_id' in child_data and child_data['user_id'] != current_user.id:
            logger.warning(f"Tentativa de privilege escalation: usuário {current_user.username} tentou especificar user_id diferente")
            raise ValueError("Operação não autorizada!")
        
        # Garantir que o user_id seja do usuário atual
        child_data['user_id'] = current_user.id
    
    # Verificar se user_id do pai e filho são iguais (proteção extra)
    if parent_record.user_id != child_data.get('user_id'):
        logger.error(f"ERRO SEGURANÇA: user_id inconsistente entre pai ({parent_record.user_id}) e filho ({child_data.get('user_id')})")
        raise ValueError("Erro de consistência de dados!")
    
    return parent_record


# Registrar filtro de template importado do módulo utils
@app.template_filter('formatar_extenso')
def template_formatar_extenso(n):
    """
    Filtro de template para formatação de números em formato extenso brasileiro.

    Converte números grandes para formato legível:
    - Valores < 1000: formato decimal padrão
    - Valores >= 1000: formato em milhares (ex: "1,5 mil")
    - Valores >= 1000000: formato em milhões (ex: "2,3 milhões")

    Args:
        n (int|float): Número a ser formatado

    Returns:
        str: Número formatado em português brasileiro

    Template Usage:
        {{ valor|formatar_extenso }}

    Examples:
        1500 -> "1,5 mil"
        2500000 -> "2,5 milhões"
        850 -> "850"
    """
    return formatar_numero_extenso(n)


@app.template_filter('dia_semana_ptbr')
def dia_semana_ptbr(data):
    """
    Filtro de template para converter dia da semana para português brasileiro.

    Args:
        data (date): Data a ser convertida

    Returns:
        str: Nome do dia da semana em português

    Template Usage:
        {{ registro.data|dia_semana_ptbr }}
    """
    dias = {
        0: 'Segunda-feira',
        1: 'Terça-feira',
        2: 'Quarta-feira',
        3: 'Quinta-feira',
        4: 'Sexta-feira',
        5: 'Sábado',
        6: 'Domingo'
    }
    return dias.get(data.weekday(), '')


# DECORADORES DE SEGURANÇA
def aprovacao_required(f):
    """Decorator para verificar se usuário está aprovado"""
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

def admin_required(f):
    """Decorator para rotas que requerem privilégios administrativos"""
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
    """Decorator para rotas que requerem privilégios de Super Administrador"""
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


# FUNÇÕES DE EMAIL
def enviar_email_verificacao(email, username, token_verificacao):
    """Envia email de verificação com link seguro para novo usuário"""
    try:
        # Se não há configuração de email, apenas registra no log
        if not app.config.get('MAIL_USERNAME'):
            app.logger.info(f"Email não enviado (config não definida) para {email} - Usuário: {username}, Token: {token_verificacao[:8]}...")
            return True
        
        msg = Message(
            subject='Bem-vindo ao CropLink - Verifique seu E-mail',
            sender=app.config['MAIL_DEFAULT_SENDER'],
            recipients=[email]
        )
        
        # Corpo do email em HTML
        msg.html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #00A859, #00964d); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
                .credentials {{ background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #00A859; margin: 20px 0; }}
                .warning {{ background: #fff3e0; border: 1px solid #ff8c00; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; margin-top: 30px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌱 Bem-vindo ao CropLink!</h1>
                    <p>Sistema de Gestão Agrícola</p>
                </div>
                <div class="content">
                    <h2>Sua conta foi criada com sucesso!</h2>
                    <p>Sua solicitação de acesso ao CropLink foi recebida e suas credenciais foram geradas.</p>
                    
                    <div class="credentials">
                        <h3>✅ Verificação de E-mail Necessária</h3>
                        <p>Para garantir a segurança da sua conta, precisamos verificar seu e-mail antes de prosseguir.</p>
                        <p><strong>Clique no botão abaixo para verificar:</strong></p>
                        <div style="text-align: center; margin: 20px 0;">
                            <a href="{request.url_root}verificar-email/{token_verificacao}" 
                               style="background: linear-gradient(135deg, #00A859, #00964d); 
                                      color: white; padding: 15px 30px; text-decoration: none; 
                                      border-radius: 8px; font-weight: bold; display: inline-block;">
                                🔐 Verificar E-mail e Definir Senha
                            </a>
                        </div>
                    </div>
                    
                    <div class="warning">
                        <h4>⚠️ Importante</h4>
                        <ul>
                            <li>Este link expira em <strong>24 horas</strong></li>
                            <li>Você definirá sua senha durante a verificação</li>
                            <li>Sua conta ainda precisará de <strong>aprovação administrativa</strong> após a verificação</li>
                        </ul>
                    </div>
                    
                    <h3>🔗 Link Alternativo</h3>
                    <p>Se o botão não funcionar, copie e cole este link no navegador:</p>
                    <p style="word-break: break-all; background: #f5f5f5; padding: 10px; border-radius: 4px;">
                        {request.url_root}verificar-email/{token_verificacao}
                    </p>
                    
                    <h3>📞 Suporte</h3>
                    <p>Em caso de dúvidas, entre em contato com nossa equipe de suporte.</p>
                </div>
                <div class="footer">
                    <p>Este é um e-mail automático do CropLink. Não responda a esta mensagem.</p>
                    <p>© 2025 CropLink - Sistema de Gestão Agrícola</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Corpo alternativo em texto simples
        msg.body = f"""
        Bem-vindo ao CropLink!
        
        Sua conta foi criada com sucesso!
        
        VERIFICAÇÃO DE E-MAIL NECESSÁRIA:
        Para garantir a segurança, precisamos verificar seu e-mail.
        
        Clique no link abaixo para verificar e definir sua senha:
        {request.url_root}verificar-email/{token_verificacao}
        
        IMPORTANTE:
        - Este link expira em 24 horas
        - Você definirá sua senha durante a verificação  
        - Sua conta ainda precisará de aprovação administrativa após a verificação
        
        © 2025 CropLink - Sistema de Gestão Agrícola
        """
        
        mail.send(msg)
        app.logger.info(f"Email de confirmação enviado para {email}")
        return True
        
    except Exception as e:
        app.logger.error(f"Erro ao enviar email para {email}: {str(e)}")
        return False


def enviar_email_reset_admin(email, nome_usuario, token_reset):
    """Envia email de reset de senha solicitado pelo administrador"""
    try:
        # Se não há configuração de email, apenas registra no log
        if not app.config.get('MAIL_USERNAME'):
            app.logger.info(f"Email de reset não enviado (config não definida) para {email} - Token: {token_reset[:8]}...")
            return True
        
        msg = Message(
            subject='CropLink - Reset de Senha Solicitado pelo Administrador',
            sender=app.config['MAIL_DEFAULT_SENDER'],
            recipients=[email]
        )
        
        # Corpo do email em HTML
        msg.html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #ff6b35, #f7931e); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
                .credentials {{ background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #ff6b35; margin: 20px 0; }}
                .warning {{ background: #fff3e0; border: 1px solid #ff8c00; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; margin-top: 30px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔒 Reset de Senha</h1>
                    <p>CropLink - Sistema de Gestão Agrícola</p>
                </div>
                <div class="content">
                    <h2>Olá, {nome_usuario}!</h2>
                    <p>Um administrador solicitou o reset da sua senha no CropLink.</p>
                    
                    <div class="credentials">
                        <h3>🔐 Redefinir Senha</h3>
                        <p>Para sua segurança, você precisa definir uma nova senha clicando no link abaixo:</p>
                        <div style="text-align: center; margin: 20px 0;">
                            <a href="{request.url_root}verificar-email/{token_reset}" 
                               style="background: linear-gradient(135deg, #ff6b35, #f7931e); 
                                      color: white; padding: 15px 30px; text-decoration: none; 
                                      border-radius: 8px; font-weight: bold; display: inline-block;">
                                🔑 Definir Nova Senha
                            </a>
                        </div>
                    </div>
                    
                    <div class="warning">
                        <h4>⚠️ Importante</h4>
                        <ul>
                            <li>Este link expira em <strong>24 horas</strong></li>
                            <li>Você será obrigado a criar uma nova senha segura</li>
                            <li>Sua conta pode precisar de nova aprovação administrativa</li>
                            <li>Se você não solicitou este reset, entre em contato com o administrador</li>
                        </ul>
                    </div>
                    
                    <h3>🔗 Link Alternativo</h3>
                    <p>Se o botão não funcionar, copie e cole este link no navegador:</p>
                    <p style="word-break: break-all; background: #f5f5f5; padding: 10px; border-radius: 4px;">
                        {request.url_root}verificar-email/{token_reset}
                    </p>
                    
                    <h3>📞 Suporte</h3>
                    <p>Em caso de dúvidas, entre em contato com nossa equipe de suporte.</p>
                </div>
                <div class="footer">
                    <p>Este é um e-mail automático do CropLink. Não responda a esta mensagem.</p>
                    <p>© 2025 CropLink - Sistema de Gestão Agrícola</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Corpo alternativo em texto simples
        msg.body = f"""
        CropLink - Reset de Senha
        
        Olá, {nome_usuario}!
        
        Um administrador solicitou o reset da sua senha no CropLink.
        
        REDEFINIR SENHA:
        Para sua segurança, você precisa definir uma nova senha.
        
        Clique no link abaixo para redefinir:
        {request.url_root}verificar-email/{token_reset}
        
        IMPORTANTE:
        - Este link expira em 24 horas
        - Você será obrigado a criar uma nova senha segura
        - Sua conta pode precisar de nova aprovação administrativa
        - Se você não solicitou este reset, entre em contato com o administrador
        
        © 2025 CropLink - Sistema de Gestão Agrícola
        """
        
        mail.send(msg)
        app.logger.info(f"Email de reset de senha enviado para {email}")
        return True
        
    except Exception as e:
        app.logger.error(f"Erro ao enviar email de reset para {email}: {str(e)}")
        return False


# ROTAS DE AUTENTICAÇÃO
# Health check endpoint - doesn't require authentication
@app.route('/health')
def health_check():
    """
    Endpoint de verificação de saúde para plataformas de deployment.
    
    Realiza verificações críticas do sistema:
    1. Conectividade com banco de dados
    2. Status geral da aplicação
    
    Usado por:
    - Load balancers para health checks
    - Monitoramento de infraestrutura
    - Verificações de deploy automático
    
    Returns:
        tuple: (response_dict, status_code)
               - 200: Sistema saudável com {'status': 'healthy', 'database': 'connected'}
               - 503: Sistema com problemas com {'status': 'unhealthy', 'error': 'description'}
               
    HTTP Methods: GET
    Content-Type: application/json
    
    Example Response (Healthy):
        {
            "status": "healthy",
            "database": "connected"
        }
    """
    try:
        # Test database connection
        from sqlalchemy import text
        db.session.execute(text('SELECT 1'))
        db.session.commit()
        return {'status': 'healthy', 'database': 'connected'}, 200
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': 'database connection failed'
        }, 503


@app.route('/')
def home():
    """
    Rota raiz que redireciona usuários baseado no status de autenticação.
    
    Implementa lógica de redirecionamento inteligente:
    - Usuários autenticados: redirecionados para dashboard
    - Usuários não autenticados: redirecionados para login
    
    Returns:
        redirect: Para '/dashboard' se autenticado, '/login' se não autenticado
        
    HTTP Methods: GET
    """
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username_or_email = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        app.logger.info(f"Tentativa de login para: {username_or_email}")

        if not username_or_email or not password:
            flash('Usuário/Email e senha são obrigatórios!', 'danger')
            return render_template('login.html')

        # Aceitar tanto username quanto email
        user = Usuario.query.filter(
            (Usuario.username == username_or_email) | (Usuario.email == username_or_email.lower())
        ).first()
        if user:
            app.logger.info(f"Usuário encontrado: {user.username}")
            password_valid = bcrypt.check_password_hash(user.password_hash, password)
            app.logger.info(f"Senha válida: {password_valid}")
            
            if password_valid:
                try:
                    # Verificar se o usuário foi aprovado
                    if not user.esta_aprovado():
                        if user.status_aprovacao == 'pendente':
                            flash('Seu cadastro está aguardando aprovação do administrador.', 'warning')
                        elif user.status_aprovacao == 'rejeitado':
                            flash('Seu cadastro foi rejeitado. Entre em contato com o administrador.', 'danger')
                        else:
                            flash('Seu acesso não está ativo. Entre em contato com o administrador.', 'danger')
                        return render_template('login.html')

                    login_user(user, remember=True)
                    app.logger.info(f"✅ Login realizado com sucesso para: {username_or_email}")

                    # Verificar se é primeiro acesso e precisa trocar senha
                    try:
                        precisa_trocar = user.precisa_trocar_senha()
                        app.logger.info(f"🔑 precisa_trocar_senha(): {precisa_trocar}")
                        if precisa_trocar:
                            flash('Por segurança, você deve alterar sua senha padrão no primeiro acesso.', 'warning')
                            return redirect(url_for('trocar_senha_primeiro_acesso'))
                    except Exception as e:
                        app.logger.error(f"❌ Erro em precisa_trocar_senha: {e}")
                        # Continuar mesmo com erro

                    # Atualizar data do último login
                    try:
                        user.data_ultimo_login = datetime.utcnow()
                        db.session.commit()
                        app.logger.info(f"✅ data_ultimo_login atualizada")
                    except Exception as e:
                        app.logger.error(f"❌ Erro ao atualizar data_ultimo_login: {e}")
                        db.session.rollback()

                    # Usar nome_completo se disponível, senão usar nome antigo ou username
                    nome_exibir = user.nome_completo or user.nome or username_or_email
                    flash(f'Bem-vindo, {nome_exibir}!', 'success')

                    app.logger.info(f"🔄 Redirecionando para dashboard...")
                    next_page = request.args.get('next')
                    return redirect(next_page) if next_page else redirect(url_for('dashboard'))

                except Exception as e:
                    app.logger.error(f"🔥 ERRO CRÍTICO durante login: {type(e).__name__}: {str(e)}")
                    import traceback
                    app.logger.error(traceback.format_exc())
                    flash(f'ERRO NO LOGIN: {type(e).__name__}: {str(e)[:300]}', 'danger')
                    flash('Tire um print e envie ao desenvolvedor', 'warning')
                    return render_template('login.html')
            else:
                flash('Senha incorreta!', 'danger')
        else:
            app.logger.warning(f"Usuário não encontrado: {username}")
            flash('Usuário não encontrado!', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Rota DESABILITADA - Registro não está mais disponível"""
    # Registro público desabilitado - apenas administradores podem criar usuários
    flash('Registro público não está disponível. Entre em contato com o administrador.', 'warning')
    return redirect(url_for('login'))

# Rota antiga removida por segurança - apenas administradores podem criar usuários


@app.route('/verificar-email/<token>', methods=['GET', 'POST'])
def verificar_email(token):
    """Rota para verificação de email e definição de senha"""
    usuario = Usuario.query.filter_by(token_verificacao=token).first()
    
    if not usuario:
        flash('Link de verificação inválido ou expirado.', 'danger')
        return redirect(url_for('register'))
    
    if not usuario.token_valido():
        flash('Link de verificação expirado. Solicite um novo cadastro.', 'danger')
        return redirect(url_for('register'))
    
    if usuario.email_verificado:
        flash('E-mail já foi verificado. Faça login normalmente.', 'info')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        nova_senha = request.form.get('nova_senha', '')
        confirmar_senha = request.form.get('confirmar_senha', '')
        
        # Validações
        errors = []
        
        if not nova_senha or len(nova_senha) < 8:
            errors.append('A senha deve ter pelo menos 8 caracteres.')
            
        if nova_senha != confirmar_senha:
            errors.append('As senhas não coincidem.')
            
        # Verificar força da senha
        if nova_senha and not any(c.isupper() for c in nova_senha):
            errors.append('A senha deve conter pelo menos uma letra maiúscula.')
            
        if nova_senha and not any(c.islower() for c in nova_senha):
            errors.append('A senha deve conter pelo menos uma letra minúscula.')
            
        if nova_senha and not any(c.isdigit() for c in nova_senha):
            errors.append('A senha deve conter pelo menos um número.')
            
        if nova_senha and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in nova_senha):
            errors.append('A senha deve conter pelo menos um caractere especial.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('verificar_email.html', token=token, usuario=usuario)
        
        try:
            # Definir nova senha e marcar email como verificado
            usuario.password_hash = bcrypt.generate_password_hash(nova_senha).decode('utf-8')
            usuario.marcar_email_verificado()
            db.session.commit()
            
            app.logger.info(f"Email verificado e senha definida para: {usuario.username}")
            flash('E-mail verificado e senha definida com sucesso! Aguarde aprovação do administrador.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Erro ao verificar email: {str(e)}")
            flash('Erro interno. Tente novamente mais tarde.', 'error')
            return render_template('verificar_email.html', token=token, usuario=usuario)
    
    return render_template('verificar_email.html', token=token, usuario=usuario)


@app.route('/trocar-senha-primeiro-acesso', methods=['GET', 'POST'])
@login_required
def trocar_senha_primeiro_acesso():
    """Rota para troca obrigatória de senha no primeiro acesso"""
    # Verificar se realmente precisa trocar senha
    if not current_user.precisa_trocar_senha():
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        nova_senha = request.form.get('nova_senha', '')
        confirmar_senha = request.form.get('confirmar_senha', '')
        
        # Validações
        errors = []
        
        if not nova_senha or len(nova_senha) < 8:
            errors.append('A nova senha deve ter pelo menos 8 caracteres.')
            
        if nova_senha == "Alterar2025#@":
            errors.append('Você não pode usar a senha padrão. Escolha uma senha diferente.')
            
        if nova_senha != confirmar_senha:
            errors.append('As senhas não coincidem.')
            
        # Verificar força da senha
        if nova_senha and not any(c.isupper() for c in nova_senha):
            errors.append('A senha deve conter pelo menos uma letra maiúscula.')
            
        if nova_senha and not any(c.islower() for c in nova_senha):
            errors.append('A senha deve conter pelo menos uma letra minúscula.')
            
        if nova_senha and not any(c.isdigit() for c in nova_senha):
            errors.append('A senha deve conter pelo menos um número.')
            
        if nova_senha and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in nova_senha):
            errors.append('A senha deve conter pelo menos um caractere especial.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('trocar_senha_primeiro_acesso.html')
        
        try:
            # Atualizar senha e marcar primeiro acesso como concluído
            current_user.password_hash = bcrypt.generate_password_hash(nova_senha).decode('utf-8')
            current_user.marcar_primeiro_acesso_concluido()
            db.session.commit()
            
            app.logger.info(f"Senha alterada com sucesso no primeiro acesso: {current_user.username}")
            flash('Senha alterada com sucesso! Bem-vindo ao CropLink!', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Erro ao alterar senha no primeiro acesso: {str(e)}")
            flash('Erro interno. Tente novamente mais tarde.', 'error')
            return render_template('trocar_senha_primeiro_acesso.html')
    
    return render_template('trocar_senha_primeiro_acesso.html')




@app.route('/logout')
@login_required
def logout():
    """
    Rota para logout de usuários autenticados.

    Executa logout seguro do usuário atual:
    1. Remove sessão do Flask-Login
    2. Exibe mensagem de confirmação
    3. Redireciona para página de login

    Returns:
        redirect: Para '/login' com mensagem de sucesso

    HTTP Methods: GET
    Security: Requer usuário autenticado (@login_required)
    """
    logout_user()
    flash('Você saiu do sistema.', 'success')
    return redirect(url_for('login'))


# ROTAS DE PERFIL E SENHA
@app.route('/meu-perfil')
@login_required
def meu_perfil():
    """Página de perfil do usuário logado"""
    return render_template('meu_perfil.html')


@app.route('/meu-perfil/atualizar', methods=['POST'])
@login_required
def atualizar_perfil():
    """Atualizar dados do perfil do usuário"""
    try:
        current_user.nome_completo = request.form.get('nome_completo', '').strip()
        email_novo = request.form.get('email', '').strip().lower()

        # Validar email único (exceto o próprio usuário)
        if email_novo != current_user.email:
            email_existe = Usuario.query.filter(
                Usuario.email == email_novo,
                Usuario.id != current_user.id
            ).first()

            if email_existe:
                flash('Este email já está em uso por outro usuário.', 'error')
                return redirect(url_for('meu_perfil'))

            current_user.email = email_novo
            current_user.email_verificado = False  # Requerer nova verificação

        db.session.commit()
        flash('Perfil atualizado com sucesso!', 'success')

    except Exception as e:
        db.session.rollback()
        flash('Erro ao atualizar perfil. Tente novamente.', 'error')
        app.logger.error(f"Erro ao atualizar perfil: {str(e)}")

    return redirect(url_for('meu_perfil'))


@app.route('/alterar-senha', methods=['POST'])
@login_required
def alterar_senha():
    """Alterar senha do usuário logado"""
    try:
        senha_atual = request.form.get('senha_atual', '')
        senha_nova = request.form.get('senha_nova', '')
        senha_confirmacao = request.form.get('senha_confirmacao', '')

        # Validar senha atual
        if not bcrypt.check_password_hash(current_user.password_hash, senha_atual):
            flash('Senha atual incorreta!', 'error')
            return redirect(url_for('meu_perfil'))

        # Validar nova senha
        if len(senha_nova) < 6:
            flash('A nova senha deve ter pelo menos 6 caracteres!', 'error')
            return redirect(url_for('meu_perfil'))

        # Validar confirmação
        if senha_nova != senha_confirmacao:
            flash('A nova senha e a confirmação não coincidem!', 'error')
            return redirect(url_for('meu_perfil'))

        # Atualizar senha
        current_user.password_hash = bcrypt.generate_password_hash(senha_nova).decode('utf-8')
        current_user.primeiro_acesso = False
        db.session.commit()

        flash('Senha alterada com sucesso!', 'success')
        app.logger.info(f"Usuário {current_user.username} alterou a senha")

    except Exception as e:
        db.session.rollback()
        flash('Erro ao alterar senha. Tente novamente.', 'error')
        app.logger.error(f"Erro ao alterar senha: {str(e)}")

    return redirect(url_for('meu_perfil'))


@app.route('/esqueci-senha', methods=['GET', 'POST'])
def esqueci_senha():
    """Página de recuperação de senha"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()

        usuario = Usuario.query.filter_by(email=email).first()

        # Sempre mostrar mensagem de sucesso (segurança)
        flash('Se o email existir em nossa base, você receberá instruções para redefinir sua senha.', 'info')

        if usuario:
            # Gerar token de reset
            token = usuario.gerar_token_verificacao()
            db.session.commit()

            # TODO: Enviar email com link de reset
            app.logger.info(f"Token de reset gerado para {email}: {token}")

        return redirect(url_for('login'))

    return render_template('esqueci_senha.html')


@app.route('/resetar-senha/<token>', methods=['GET', 'POST'])
def resetar_senha(token):
    """Resetar senha usando token"""
    usuario = Usuario.query.filter_by(token_verificacao=token).first()

    if not usuario or not usuario.token_valido():
        flash('Link de recuperação inválido ou expirado!', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        senha_nova = request.form.get('senha_nova', '')
        senha_confirmacao = request.form.get('senha_confirmacao', '')

        if len(senha_nova) < 6:
            flash('A senha deve ter pelo menos 6 caracteres!', 'error')
            return redirect(url_for('resetar_senha', token=token))

        if senha_nova != senha_confirmacao:
            flash('As senhas não coincidem!', 'error')
            return redirect(url_for('resetar_senha', token=token))

        # Atualizar senha
        usuario.password_hash = bcrypt.generate_password_hash(senha_nova).decode('utf-8')
        usuario.token_verificacao = None
        usuario.token_expiracao = None
        usuario.primeiro_acesso = False
        db.session.commit()

        flash('Senha redefinida com sucesso! Faça login com sua nova senha.', 'success')
        return redirect(url_for('login'))

    return render_template('resetar_senha.html', token=token)


# PÁGINA DE UPGRADE DE PLANOS
@app.route('/upgrade-plano')
@login_required
def upgrade_plano():
    """Página para upgrade de planos"""
    plano_atual = current_user.plano
    planos_disponiveis = PLANOS_CONFIG
    
    return render_template('upgrade_plano.html', 
                         plano_atual=plano_atual,
                         planos=planos_disponiveis,
                         current_user=current_user)

@app.route('/teste-alterar-plano/<plano>')
@login_required
def teste_alterar_plano(plano):
    """Função de teste para alterar plano do usuário (APENAS PARA DEMONSTRAÇÃO)"""
    if plano in ['basic', 'plus', 'premium']:
        current_user.plano = plano
        db.session.commit()
        flash(f'Plano alterado para {PLANOS_CONFIG[plano]["nome"]} com sucesso!', 'success')
    else:
        flash('Plano inválido!', 'danger')
    
    return redirect(url_for('upgrade_plano'))


# DASHBOARD - FUNÇÕES AUXILIARES MODULARES

def obter_estatisticas_principais():
    """
    Obtém as estatísticas principais do sistema (contadores de entidades).
    
    Returns:
        tuple: (total_insumos, total_maquinas, total_funcionarios, total_silos, insumos_baixo_estoque_count)
    """
    total_insumos = contar_registros_usuario(Insumo)
    total_maquinas = contar_registros_usuario(Maquinario)
    total_funcionarios = contar_registros_usuario(Funcionario)
    total_silos = contar_registros_usuario(Silo)
    insumos_baixo_estoque_count = filtrar_por_usuario(
        Insumo.query.filter(Insumo.quantidade < 10), Insumo
    ).count()
    
    return (total_insumos, total_maquinas, total_funcionarios, total_silos, insumos_baixo_estoque_count)


def obter_dados_insumos_baixo_estoque():
    """
    Obtém lista de insumos com baixo estoque (< 10 unidades).
    
    Returns:
        list: Lista de objetos Insumo com baixo estoque
    """
    return filtrar_por_usuario(
        Insumo.query.filter(Insumo.quantidade < 10), Insumo
    ).limit(20).all()


def obter_movimentacoes_recentes():
    """
    Obtém as últimas movimentações de insumos do usuário.
    
    Returns:
        list: Lista de objetos MovimentacaoInsumo ordenados por data decrescente
    """
    return filtrar_por_usuario(
        db.session.query(MovimentacaoInsumo)
        .options(db.joinedload(MovimentacaoInsumo.insumo))
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
    return filtrar_por_usuario(
        RegistroChuva.query.filter(RegistroChuva.data >= data_inicio),
        RegistroChuva
    ).order_by(RegistroChuva.data.desc()).limit(10).all()


# DASHBOARD
@app.route('/dashboard')
@login_required
def dashboard():
    """
    Dashboard principal do sistema com estatísticas e gráficos em tempo real.
    
    Apresenta visão geral das operações agrícolas:
    - Estatísticas principais (insumos, máquinas, funcionários, silos)
    - Alertas de insumos com baixo estoque
    - Gráficos de movimentações dos últimos 7 dias
    - Dados de chuva e registros meteorológicos
    - Movimentações recentes do sistema
    
    Funcionalidades por perfil:
    - Admin: Visão global de todos os dados
    - Cliente: Apenas seus próprios dados
    - Funcionário: Dados do produtor rural vinculado
    
    Returns:
        rendered_template: dashboard.html com contexto completo
        
    Template Variables:
        - estatisticas: Contadores principais
        - insumos_baixo_estoque: Lista de alertas
        - dados_graficos: Dados para Charts.js
        - movimentacoes_recentes: Últimas 10 movimentações
        - registros_chuva: Dados meteorológicos
        
    HTTP Methods: GET
    Security: Requer usuário autenticado e aprovado
    """
    """
    Rota principal do dashboard que agrega dados de múltiplas fontes.
    
    Returns:
        Response: Template renderizado com dados do dashboard ou página de erro
    """
    try:
        data_inicio = datetime.utcnow() - timedelta(days=7)

        # Coletar dados usando funções modulares especializadas
        (total_insumos, total_maquinas, total_funcionarios, 
         total_silos, insumos_baixo_estoque_count) = obter_estatisticas_principais()
        
        insumos_baixo_estoque = obter_dados_insumos_baixo_estoque()
        movimentacoes_recentes = obter_movimentacoes_recentes()
        
        # Processar dados de movimentações para gráficos
        movimentacoes_agregadas = obter_agregacao_movimentacoes(data_inicio)
        (labels_movimentacoes, dados_entradas, 
         dados_saidas) = processar_dados_movimentacoes_grafico(movimentacoes_agregadas)
        
        # Processar dados de chuva para gráficos
        chuva_agregada = obter_agregacao_chuva(data_inicio)
        (total_chuva_semana, media_chuva_semana, 
         labels_chuva, dados_chuva) = processar_dados_chuva_grafico(chuva_agregada)
        
        registros_chuva = obter_registros_chuva_recentes(data_inicio)

        return render_template(
            'dashboard.html',
            total_insumos=total_insumos,
            total_maquinas=total_maquinas,
            total_funcionarios=total_funcionarios,
            total_silos=total_silos,
            insumos_baixo_estoque=insumos_baixo_estoque,
            movimentacoes_recentes=movimentacoes_recentes,
            labels_movimentacoes=json.dumps(labels_movimentacoes),
            dados_entradas=json.dumps(dados_entradas),
            dados_saidas=json.dumps(dados_saidas),
            registros_chuva=registros_chuva,
            total_chuva_semana=total_chuva_semana,
            media_chuva_semana=media_chuva_semana,
            labels_chuva=json.dumps(labels_chuva),
            dados_chuva=json.dumps(dados_chuva))

    except Exception as e:
        app.logger.error(f"Erro no dashboard: {str(e)}")
        flash('Erro ao carregar dashboard. Tente novamente.', 'error')

        # Dados de fallback em caso de erro
        return render_template('dashboard.html',
                               total_insumos=0,
                               total_maquinas=0,
                               total_funcionarios=0,
                               total_silos=0,
                               insumos_baixo_estoque=[],
                               movimentacoes_recentes=[],
                               labels_movimentacoes=json.dumps(['Sem dados']),
                               dados_entradas=json.dumps([0]),
                               dados_saidas=json.dumps([0]),
                               registros_chuva=[],
                               total_chuva_semana=0,
                               media_chuva_semana=0,
                               labels_chuva=json.dumps(['Sem dados']),
                               dados_chuva=json.dumps([0]))


# ROTAS DE INSUMOS
@app.route('/insumos')
@login_required
def insumos():
    insumos_list = filtrar_por_usuario(Insumo.query, Insumo).all()
    return render_template('insumos.html', insumos=insumos_list)


@app.route('/insumos/adicionar', methods=['POST'])
@login_required
def adicionar_insumo():
    nome = request.form.get('nome')
    quantidade = float(request.form.get('quantidade', 0))
    unidade = request.form.get('unidade')

    insumo = criar_com_usuario(Insumo, nome=nome, quantidade=quantidade, unidade=unidade)
    db.session.add(insumo)
    db.session.commit()

    flash('Insumo adicionado com sucesso!', 'success')
    return redirect(url_for('insumos'))


@app.route('/insumos/movimentar/<int:id>', methods=['POST'])
@login_required
def movimentar_insumo(id):
    # CORREÇÃO DE SEGURANÇA: Verificar ownership do insumo antes de movimentar
    insumo = obter_por_id_usuario(Insumo, id)
    if not insumo:
        flash('Insumo não encontrado ou acesso negado!', 'error')
        return redirect(url_for('insumos'))

    tipo = request.form.get('tipo')
    quantidade = float(request.form.get('quantidade', 0))
    observacao = request.form.get('observacao', '')

    # SEGURANÇA: Validar ownership pai-filho para prevenir cross-user attacks
    movimentacao_data = {
        'tipo': tipo,
        'quantidade': quantidade,
        'observacao': observacao,
        'insumo_id': id,
        'user_id': current_user.id  # CORREÇÃO: Adicionar user_id para prevenir erro de inconsistência
    }
    
    try:
        insumo = validate_parent_child_ownership(Insumo, id, MovimentacaoInsumo, movimentacao_data)
        movimentacao = criar_com_usuario(MovimentacaoInsumo, **movimentacao_data)
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('insumos'))
    db.session.add(movimentacao)

    # Atualizar estoque
    if tipo == 'Entrada':
        insumo.quantidade += quantidade
        flash('Estoque adicionado com sucesso!', 'success')
    else:  # Saída
        if insumo.quantidade < quantidade:
            flash('Quantidade insuficiente em estoque!', 'error')
            return redirect(url_for('insumos'))
        insumo.quantidade -= quantidade
        flash('Baixa no estoque registrada com sucesso!', 'success')

    db.session.commit()
    return redirect(url_for('insumos'))


@app.route('/insumos/editar/<int:id>', methods=['POST'])
@login_required
def editar_insumo(id):
    """Rota para editar insumos gerais com validação de ownership"""
    insumo = obter_por_id_usuario(Insumo, id)
    if not insumo:
        flash('Insumo não encontrado ou acesso negado!', 'error')
        return redirect(url_for('insumos'))

    insumo.nome = request.form.get('nome')
    insumo.quantidade = float(request.form.get('quantidade', 0))
    insumo.unidade = request.form.get('unidade')

    db.session.commit()
    flash('Insumo atualizado com sucesso!', 'success')
    return redirect(url_for('insumos'))


@app.route('/insumos/excluir/<int:id>', methods=['POST'])
@login_required
def excluir_insumo(id):
    """Rota para excluir insumos gerais com validação de ownership"""
    insumo = obter_por_id_usuario(Insumo, id)
    if not insumo:
        flash('Insumo não encontrado ou acesso negado!', 'error')
        return redirect(url_for('insumos'))
    
    db.session.delete(insumo)
    db.session.commit()
    flash('Insumo excluído com sucesso!', 'success')
    return redirect(url_for('insumos'))


# ROTAS DE INSUMOS AGRÍCOLAS
@app.route('/insumos-agricolas')
@login_required
def insumos_agricolas():
    insumos_list = filtrar_por_usuario(InsumoAgricola.query, InsumoAgricola).all()
    return render_template('insumos_agricolas.html', insumos=insumos_list)


@app.route('/insumos-agricolas/adicionar', methods=['POST'])
@login_required
def adicionar_insumo_agricola():
    nome = request.form.get('nome')
    quantidade = float(request.form.get('quantidade', 0))
    unidade = request.form.get('unidade')
    categoria = request.form.get('categoria')
    observacao = request.form.get('observacao', '')

    insumo = criar_com_usuario(InsumoAgricola, nome=nome,
                            quantidade=quantidade,
                            unidade=unidade,
                            categoria=categoria,
                            observacao=observacao)
    db.session.add(insumo)
    db.session.commit()

    flash('Insumo agrícola adicionado com sucesso!', 'success')
    return redirect(url_for('insumos_agricolas'))


@app.route('/insumos-agricolas/editar/<int:id>', methods=['POST'])
@login_required
def editar_insumo_agricola(id):
    insumo = obter_por_id_usuario(InsumoAgricola, id)
    if not insumo:
        flash('Insumo não encontrado ou acesso negado!', 'error')
        return redirect(url_for('insumos_agricolas'))

    insumo.nome = request.form.get('nome')
    insumo.quantidade = float(request.form.get('quantidade', 0))
    insumo.unidade = request.form.get('unidade')
    insumo.categoria = request.form.get('categoria')
    insumo.observacao = request.form.get('observacao', '')

    db.session.commit()
    flash('Insumo agrícola atualizado com sucesso!', 'success')
    return redirect(url_for('insumos_agricolas'))


@app.route('/insumos-agricolas/excluir/<int:id>', methods=['POST'])
@login_required
def excluir_insumo_agricola(id):
    insumo = obter_por_id_usuario(InsumoAgricola, id)
    if not insumo:
        flash('Insumo não encontrado ou acesso negado!', 'error')
        return redirect(url_for('insumos_agricolas'))
    
    # Verificar se há aplicações do insumo registradas
    aplicacoes = filtrar_por_usuario(
        AplicacaoInsumo.query.filter_by(insumo_agricola_id=id),
        AplicacaoInsumo).count()
    
    if aplicacoes > 0:
        flash(
            f'Não é possível deletar o insumo "{insumo.nome}" porque ele possui aplicações registradas!',
            'error')
        return redirect(url_for('insumos_agricolas'))
    
    nome_insumo = insumo.nome
    db.session.delete(insumo)
    db.session.commit()
    flash(f'Insumo agrícola "{nome_insumo}" excluído com sucesso!', 'success')
    return redirect(url_for('insumos_agricolas'))


@app.route('/insumos-agricolas/movimentar/<int:id>', methods=['POST'])
@login_required
def movimentar_insumo_agricola(id):
    """Rota para movimentar insumos agrícolas (aplicação, entrada de estoque, baixa)"""
    # CORREÇÃO DE SEGURANÇA: Verificar ownership do insumo antes de movimentar
    insumo = obter_por_id_usuario(InsumoAgricola, id)
    if not insumo:
        flash('Insumo agrícola não encontrado ou acesso negado!', 'error')
        return redirect(url_for('insumos_agricolas'))

    tipo = request.form.get('tipo')
    quantidade = float(request.form.get('quantidade', 0))
    observacao = request.form.get('observacao', '')
    
    # Campos específicos para aplicação no campo
    talhao = request.form.get('talhao', '')
    dose_aplicada = request.form.get('dose_aplicada')
    unidade_dose = request.form.get('unidade_dose', '')
    condicao_aplicacao = request.form.get('condicao_aplicacao', '')
    
    # Campo motivo da baixa (combinar com observação)
    motivo = request.form.get('motivo', '')
    if motivo and observacao:
        observacao_completa = f"Motivo: {motivo}. {observacao}"
    elif motivo:
        observacao_completa = f"Motivo: {motivo}"
    else:
        observacao_completa = observacao

    # Convertir dose_aplicada para float se preenchida
    dose_aplicada_float = float(dose_aplicada) if dose_aplicada else None

    # SEGURANÇA: Validar ownership pai-filho para prevenir cross-user attacks
    movimentacao_data = {
        'tipo': tipo,
        'quantidade': quantidade,
        'observacao': observacao_completa,
        'talhao': talhao,
        'dose_aplicada': dose_aplicada_float,
        'unidade_dose': unidade_dose,
        'condicao_aplicacao': condicao_aplicacao,
        'insumo_agricola_id': id,
        'user_id': current_user.id  # CORREÇÃO: Adicionar user_id para prevenir erro de inconsistência
    }
    
    try:
        insumo = validate_parent_child_ownership(InsumoAgricola, id, MovimentacaoInsumoAgricola, movimentacao_data)
        movimentacao = criar_com_usuario(MovimentacaoInsumoAgricola, **movimentacao_data)
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('insumos_agricolas'))
    
    db.session.add(movimentacao)

    # Atualizar estoque
    if tipo == 'Entrada':
        insumo.quantidade += quantidade
        flash('Estoque adicionado com sucesso!', 'success')
    else:  # Saída
        if insumo.quantidade < quantidade:
            flash('Quantidade insuficiente em estoque!', 'error')
            return redirect(url_for('insumos_agricolas'))
        insumo.quantidade -= quantidade
        if talhao:  # É uma aplicação no campo
            flash(f'Aplicação registrada no talhão {talhao} com sucesso!', 'success')
        else:  # É uma baixa de estoque
            flash('Baixa no estoque registrada com sucesso!', 'success')

    db.session.commit()
    return redirect(url_for('insumos_agricolas'))


# ROTAS DE APLICAÇÃO DE INSUMOS
@app.route('/aplicacao-insumos')
@login_required
def aplicacao_insumos():
    """Tela principal para aplicação de insumos agrícolas"""
    # SEGURANÇA: Usar contexto de produtor rural para carregamento
    ctx_produtor = current_user.get_produtor_contexto()
    
    if ctx_produtor:
        # Cliente ou funcionário - buscar insumos do contexto do produtor rural
        insumos_list = db.session.query(InsumoAgricola).join(Usuario).filter(
            Usuario.produtor_rural_id == ctx_produtor
        ).all()
        
        # Buscar histórico de aplicações do contexto do produtor rural
        aplicacoes = db.session.query(AplicacaoInsumo).join(InsumoAgricola).join(Usuario).filter(
            Usuario.produtor_rural_id == ctx_produtor
        ).order_by(AplicacaoInsumo.data_aplicacao.desc()).limit(10).all()
    elif current_user.user_role == 'super_admin':
        # Super admin - acesso total
        insumos_list = InsumoAgricola.query.all()
        aplicacoes = AplicacaoInsumo.query.order_by(AplicacaoInsumo.data_aplicacao.desc()).limit(10).all()
    else:
        # Fallback - filtrar por user_id
        insumos_list = filtrar_por_usuario(InsumoAgricola.query, InsumoAgricola).all()
        aplicacoes = filtrar_por_usuario(AplicacaoInsumo.query, AplicacaoInsumo).order_by(AplicacaoInsumo.data_aplicacao.desc()).limit(10).all()
    
    return render_template('aplicacao_insumos.html', 
                         insumos=insumos_list, 
                         aplicacoes=aplicacoes)

@app.route('/aplicacao-insumos/aplicar', methods=['POST'])
@login_required
def aplicar_insumo():
    """Processar aplicação de múltiplos insumos agrícolas com segurança multi-tenant e transações atômicas"""
    import json
    from collections import defaultdict
    
    # Receber dados de múltiplos insumos
    insumos_json = request.form.get('insumos_aplicacao')
    talhao = request.form.get('talhao', '').strip()
    observacao = request.form.get('observacao', '').strip()
    
    # Validação básica de entrada
    if not insumos_json:
        flash('Nenhum insumo selecionado para aplicação!', 'error')
        return redirect(url_for('aplicacao_insumos'))
    
    try:
        insumos_dados = json.loads(insumos_json)
    except json.JSONDecodeError:
        flash('Dados de insumos inválidos!', 'error')
        return redirect(url_for('aplicacao_insumos'))
    
    if not insumos_dados or len(insumos_dados) == 0:
        flash('Adicione pelo menos um insumo para aplicação!', 'error')
        return redirect(url_for('aplicacao_insumos'))
    
    # TRANSAÇÃO ATÔMICA COM ROW LOCKING para todos os insumos
    try:
        # SEGURANÇA: Obter contexto do produtor rural para validação multi-tenant
        ctx_produtor = current_user.get_produtor_contexto()
        
        # FASE 1: Validação e deduplicação de insumos
        insumos_agrupados = defaultdict(float)  # {insumo_id: quantidade_total}
        
        for item in insumos_dados:
            # Validação robusta de dados
            try:
                insumo_id = int(item.get('id'))
                quantidade_aplicada = float(item.get('quantidade_aplicada', 0))
            except (ValueError, TypeError):
                flash(f'Dados inválidos para o insumo {item.get("nome", "desconhecido")}!', 'error')
                return redirect(url_for('aplicacao_insumos'))
            
            if quantidade_aplicada <= 0:
                flash(f'Quantidade deve ser maior que zero para {item.get("nome", "desconhecido")}!', 'error')
                return redirect(url_for('aplicacao_insumos'))
            
            # Agrupar quantidades por insumo (deduplica automaticamente)
            insumos_agrupados[insumo_id] += quantidade_aplicada
        
        # FASE 2: Buscar e bloquear insumos com validação de segurança
        insumos_locked = []
        aplicacoes_para_criar = []
        movimentacoes_para_criar = []
        
        for insumo_id, quantidade_total in insumos_agrupados.items():
            # Buscar insumo com ROW LOCK e validação multi-tenant
            if ctx_produtor:
                # Cliente ou funcionário - usar contexto do produtor rural
                insumo = db.session.query(InsumoAgricola).join(Usuario).filter(
                    InsumoAgricola.id == insumo_id,
                    Usuario.produtor_rural_id == ctx_produtor
                ).with_for_update().first()
            elif current_user.user_role == 'super_admin':
                # Super admin - acesso total a qualquer insumo
                insumo = db.session.query(InsumoAgricola).filter_by(
                    id=insumo_id
                ).with_for_update().first()
            else:
                # Fallback - usar filtro direto por user_id
                insumo = db.session.query(InsumoAgricola).filter_by(
                    id=insumo_id, 
                    user_id=current_user.id
                ).with_for_update().first()
            
            if not insumo:
                flash(f'Insumo com ID {insumo_id} não encontrado ou acesso negado!', 'error')
                return redirect(url_for('aplicacao_insumos'))
            
            # Verificar estoque após o lock (critical section)
            if insumo.quantidade < quantidade_total:
                flash(f'Estoque insuficiente para {insumo.nome}! Disponível: {insumo.quantidade} {insumo.unidade}, Solicitado: {quantidade_total}', 'error')
                return redirect(url_for('aplicacao_insumos'))
            
            insumos_locked.append((insumo, quantidade_total))
            
            # Preparar dados da aplicação com validação multi-tenant
            aplicacao_data = {
                'insumo_agricola_id': insumo.id,
                'quantidade_aplicada': quantidade_total,
                'talhao': talhao,
                'observacao': observacao,
                'user_id': current_user.id
            }
            
            aplicacao = criar_com_usuario(AplicacaoInsumo, **aplicacao_data)
            aplicacoes_para_criar.append(aplicacao)
            
            # Preparar movimentação de estoque para auditoria
            movimentacao_data = {
                'tipo': 'Saída',
                'quantidade': quantidade_total,
                'observacao': f'Aplicação no campo - {observacao if observacao else "Aplicação de insumo"}',
                'talhao': talhao,
                'condicao_aplicacao': 'Campo',
                'insumo_agricola_id': insumo.id,
                'user_id': current_user.id
            }
            
            movimentacao = criar_com_usuario(MovimentacaoInsumoAgricola, **movimentacao_data)
            movimentacoes_para_criar.append(movimentacao)
        
        # FASE 3: Aplicar todas as mudanças atomicamente
        insumos_processados = []
        
        for (insumo, quantidade_total), aplicacao, movimentacao in zip(insumos_locked, aplicacoes_para_criar, movimentacoes_para_criar):
            # Atualizar estoque
            estoque_anterior = insumo.quantidade
            insumo.quantidade -= quantidade_total
            
            # Log da operação
            print(f"APLICAÇÃO MÚLTIPLA SEGURA: {insumo.nome} - Anterior: {estoque_anterior}, Aplicado: {quantidade_total}, Novo: {insumo.quantidade}")
            
            # Adicionar à sessão
            db.session.add(aplicacao)
            db.session.add(movimentacao)
            insumos_processados.append(insumo.nome)
        
        # COMMIT MANUAL
        db.session.commit()
        
        # Mensagem de sucesso após commit bem-sucedido
        total_insumos = len(insumos_processados)
        if total_insumos == 1:
            if talhao:
                flash(f'Aplicação de {insumos_processados[0]} realizada no talhão {talhao}!', 'success')
            else:
                flash(f'Aplicação de {insumos_processados[0]} registrada com sucesso!', 'success')
        else:
            insumos_lista = ', '.join(insumos_processados)
            if talhao:
                flash(f'Aplicação de {total_insumos} insumos ({insumos_lista}) realizada no talhão {talhao}!', 'success')
            else:
                flash(f'Aplicação de {total_insumos} insumos ({insumos_lista}) registrada com sucesso!', 'success')
                
    except ValueError as e:
        db.session.rollback()
        print(f"ERRO ValueError na aplicação múltipla: {str(e)}")
        flash(str(e), 'error')
    except Exception as e:
        db.session.rollback()
        print(f"ERRO Exception na aplicação múltipla: {str(e)} - Tipo: {type(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Erro ao registrar aplicação: {str(e)}', 'error')
    
    return redirect(url_for('aplicacao_insumos'))


@app.route('/api/insumos-agricolas/<int:id>')
@login_required
def api_insumo_agricola(id):
    """API para buscar dados de insumo agrícola em tempo real"""
    insumo = obter_por_id_usuario(InsumoAgricola, id)
    if not insumo:
        return jsonify({'error': 'Insumo não encontrado'}), 404
    
    return jsonify({
        'id': insumo.id,
        'nome': insumo.nome,
        'quantidade': insumo.quantidade,
        'unidade': insumo.unidade,
        'categoria': insumo.categoria
    })


# ROTAS DE MAQUINÁRIO
@app.route('/maquinario')
@login_required
def maquinario():
    maquinas = filtrar_por_usuario(Maquinario.query, Maquinario).all()
    return render_template('maquinario.html', maquinas=maquinas)


@app.route('/maquinario/adicionar', methods=['POST'])
@login_required
def adicionar_maquina():
    try:
        nome = request.form.get('nome')
        marca = request.form.get('marca', '')
        modelo_maquina = request.form.get('modelo', '')  # Renomeado para evitar conflito
        ano_str = request.form.get('ano', '').strip()
        ano = int(ano_str) if ano_str and ano_str.isdigit() else None
        status = request.form.get('status', 'Operacional')
        horas_str = request.form.get('horas_de_uso', '').strip()
        horas_de_uso = float(horas_str) if horas_str and horas_str.replace('.', '').isdigit() else 0
        
        # Campos de manutenção
        tipo_oleo = request.form.get('tipo_oleo', '')
        filtro_oleo = request.form.get('filtro_oleo', '')
        filtro_ar = request.form.get('filtro_ar', '')
        filtro_combustivel = request.form.get('filtro_combustivel', '')
        
        # Campos de datas das últimas trocas
        data_ultima_troca_oleo = None
        data_ultima_troca_filtro_oleo = None
        data_ultima_troca_filtro_ar = None
        data_ultima_troca_filtro_combustivel = None
        
        if request.form.get('data_ultima_troca_oleo'):
            try:
                data_ultima_troca_oleo = datetime.strptime(request.form.get('data_ultima_troca_oleo'), '%Y-%m-%d').date()
            except:
                pass
                
        if request.form.get('data_ultima_troca_filtro_oleo'):
            try:
                data_ultima_troca_filtro_oleo = datetime.strptime(request.form.get('data_ultima_troca_filtro_oleo'), '%Y-%m-%d').date()
            except:
                pass
                
        if request.form.get('data_ultima_troca_filtro_ar'):
            try:
                data_ultima_troca_filtro_ar = datetime.strptime(request.form.get('data_ultima_troca_filtro_ar'), '%Y-%m-%d').date()
            except:
                pass
                
        if request.form.get('data_ultima_troca_filtro_combustivel'):
            try:
                data_ultima_troca_filtro_combustivel = datetime.strptime(request.form.get('data_ultima_troca_filtro_combustivel'), '%Y-%m-%d').date()
            except:
                pass

        if not nome:
            flash('Nome da máquina é obrigatório!', 'error')
            return redirect(url_for('maquinario'))

        maquina = criar_com_usuario(Maquinario,
                                    nome=nome,
                                    marca=marca,
                                    modelo=modelo_maquina,  # Corrigido o conflito de nomes
                                    ano=ano,
                                    status=status,
                                    horas_de_uso=horas_de_uso,
                                    tipo_oleo=tipo_oleo,
                                    filtro_oleo=filtro_oleo,
                                    filtro_ar=filtro_ar,
                                    filtro_combustivel=filtro_combustivel,
                                    data_ultima_troca_oleo=data_ultima_troca_oleo,
                                    data_ultima_troca_filtro_oleo=data_ultima_troca_filtro_oleo,
                                    data_ultima_troca_filtro_ar=data_ultima_troca_filtro_ar,
                                    data_ultima_troca_filtro_combustivel=data_ultima_troca_filtro_combustivel)
        db.session.add(maquina)
        db.session.commit()

        flash('Máquina adicionada com sucesso!', 'success')
    except ValueError as e:
        flash('Erro nos dados fornecidos. Verifique os valores numéricos.',
              'error')
    except Exception as e:
        flash('Erro ao adicionar máquina. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar máquina: {str(e)}")

    return redirect(url_for('maquinario'))


@app.route('/maquinario/editar/<int:id>', methods=['POST'])
@login_required
def editar_maquina(id):
    try:
        # SEGURANÇA: Usar função segura de validação
        maquina = get_or_404_user_scoped(Maquinario, id)

        maquina.nome = request.form.get('nome')
        maquina.marca = request.form.get('marca', '')
        maquina.modelo = request.form.get('modelo', '')
        ano_str = request.form.get('ano', '').strip()
        maquina.ano = int(ano_str) if ano_str and ano_str.isdigit() else None
        maquina.status = request.form.get('status', 'Operacional')
        horas_str = request.form.get('horas_de_uso', '').strip()
        maquina.horas_de_uso = float(horas_str) if horas_str and horas_str.replace('.', '').isdigit() else 0
        
        # Campos de manutenção
        maquina.tipo_oleo = request.form.get('tipo_oleo', '')
        maquina.filtro_oleo = request.form.get('filtro_oleo', '')
        maquina.filtro_ar = request.form.get('filtro_ar', '')
        maquina.filtro_combustivel = request.form.get('filtro_combustivel', '')
        
        # Campos de datas das últimas trocas
        if request.form.get('data_ultima_troca_oleo'):
            try:
                maquina.data_ultima_troca_oleo = datetime.strptime(request.form.get('data_ultima_troca_oleo'), '%Y-%m-%d').date()
            except:
                pass
        else:
            maquina.data_ultima_troca_oleo = None
                
        if request.form.get('data_ultima_troca_filtro_oleo'):
            try:
                maquina.data_ultima_troca_filtro_oleo = datetime.strptime(request.form.get('data_ultima_troca_filtro_oleo'), '%Y-%m-%d').date()
            except:
                pass
        else:
            maquina.data_ultima_troca_filtro_oleo = None
                
        if request.form.get('data_ultima_troca_filtro_ar'):
            try:
                maquina.data_ultima_troca_filtro_ar = datetime.strptime(request.form.get('data_ultima_troca_filtro_ar'), '%Y-%m-%d').date()
            except:
                pass
        else:
            maquina.data_ultima_troca_filtro_ar = None
                
        if request.form.get('data_ultima_troca_filtro_combustivel'):
            try:
                maquina.data_ultima_troca_filtro_combustivel = datetime.strptime(request.form.get('data_ultima_troca_filtro_combustivel'), '%Y-%m-%d').date()
            except:
                pass
        else:
            maquina.data_ultima_troca_filtro_combustivel = None

        if not maquina.nome:
            flash('Nome da máquina é obrigatório!', 'error')
            return redirect(url_for('maquinario'))

        db.session.commit()
        flash('Máquina atualizada com sucesso!', 'success')
    except ValueError as e:
        flash('Erro nos dados fornecidos. Verifique os valores numéricos.',
              'error')
    except Exception as e:
        flash('Erro ao atualizar máquina. Tente novamente.', 'error')
        app.logger.error(f"Erro ao editar máquina: {str(e)}")

    return redirect(url_for('maquinario'))


@app.route('/maquinario/excluir/<int:id>', methods=['POST'])
@login_required
def excluir_maquina(id):
    try:
        maquina = obter_por_id_usuario(Maquinario, id)
        if not maquina:
            flash('Máquina não encontrada ou acesso negado!', 'error')
            return redirect(url_for('maquinario'))
        
        db.session.delete(maquina)
        db.session.commit()
        flash('Máquina excluída com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao excluir máquina. Tente novamente.', 'error')
        app.logger.error(f"Erro ao excluir máquina: {str(e)}")

    return redirect(url_for('maquinario'))


# ROTAS DE FUNCIONÁRIOS
@app.route('/funcionarios')
@login_required
def funcionarios():
    funcionarios_list = filtrar_por_usuario(Funcionario.query, Funcionario).all()
    return render_template('funcionarios.html', funcionarios=funcionarios_list)


@app.route('/funcionarios/adicionar', methods=['POST'])
@login_required
def adicionar_funcionario():
    try:
        nome_completo = request.form.get('nome_completo')
        cpf = request.form.get('cpf', '')
        telefone = request.form.get('telefone', '')
        cargo = request.form.get('cargo', '')
        data_admissao_str = request.form.get('data_admissao')

        if not nome_completo:
            flash('Nome completo é obrigatório!', 'error')
            return redirect(url_for('funcionarios'))

        # Verificar se CPF já existe (se fornecido)
        if cpf:
            funcionario_existente = filtrar_por_usuario(
                Funcionario.query.filter_by(cpf=cpf), Funcionario).first()
            if funcionario_existente:
                flash('CPF já está cadastrado para outro funcionário!',
                      'error')
                return redirect(url_for('funcionarios'))

        data_admissao = None
        if data_admissao_str:
            data_admissao = datetime.strptime(data_admissao_str,
                                              '%Y-%m-%d').date()

        funcionario = criar_com_usuario(Funcionario,
                                        nome_completo=nome_completo,
                                        cpf=cpf,
                                        telefone=telefone,
                                        cargo=cargo,
                                        data_admissao=data_admissao)
        db.session.add(funcionario)
        db.session.commit()

        flash('Funcionário adicionado com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Data de admissão inválida.', 'error')
    except Exception as e:
        flash('Erro ao adicionar funcionário. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar funcionário: {str(e)}")

    return redirect(url_for('funcionarios'))


@app.route('/funcionarios/editar/<int:id>', methods=['POST'])
@login_required
def editar_funcionario(id):
    try:
        # SEGURANÇA: Usar função segura de validação
        funcionario = get_or_404_user_scoped(Funcionario, id)

        nome_completo = request.form.get('nome_completo')
        cpf = request.form.get('cpf', '')

        if not nome_completo:
            flash('Nome completo é obrigatório!', 'error')
            return redirect(url_for('funcionarios'))

        # Verificar se CPF já existe para outro funcionário (se fornecido)
        if cpf:
            funcionario_existente = filtrar_por_usuario(
                Funcionario.query.filter_by(cpf=cpf).filter(Funcionario.id != id),
                Funcionario).first()
            if funcionario_existente:
                flash('CPF já está cadastrado para outro funcionário!',
                      'error')
                return redirect(url_for('funcionarios'))

        funcionario.nome_completo = nome_completo
        funcionario.cpf = cpf
        funcionario.telefone = request.form.get('telefone', '')
        funcionario.cargo = request.form.get('cargo', '')

        data_admissao_str = request.form.get('data_admissao')
        if data_admissao_str:
            funcionario.data_admissao = datetime.strptime(
                data_admissao_str, '%Y-%m-%d').date()

        db.session.commit()
        flash('Funcionário atualizado com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Data de admissão inválida.', 'error')
    except Exception as e:
        flash('Erro ao atualizar funcionário. Tente novamente.', 'error')
        app.logger.error(f"Erro ao editar funcionário: {str(e)}")

    return redirect(url_for('funcionarios'))


@app.route('/funcionarios/excluir/<int:id>', methods=['POST'])
@login_required
def excluir_funcionario(id):
    funcionario = obter_por_id_usuario(Funcionario, id)
    if not funcionario:
        flash('Funcionário não encontrado ou acesso negado!', 'error')
        return redirect(url_for('funcionarios'))
    
    db.session.delete(funcionario)
    db.session.commit()
    flash('Funcionário excluído com sucesso!', 'success')
    return redirect(url_for('funcionarios'))


# ROTAS DE DIARISTAS
@app.route('/diaristas')
@login_required
def diaristas():
    diaristas_list = filtrar_por_usuario(Diarista.query, Diarista).all()
    return render_template('diaristas.html', diaristas=diaristas_list)


@app.route('/diaristas/adicionar', methods=['POST'])
@login_required
def adicionar_diarista():
    try:
        nome_completo = request.form.get('nome_completo')
        cpf = request.form.get('cpf', '')
        telefone = request.form.get('telefone', '')
        valor_diaria = float(request.form.get('valor_diaria', 0))

        if not nome_completo:
            flash('Nome completo é obrigatório!', 'error')
            return redirect(url_for('diaristas'))

        if valor_diaria < 0:
            flash('Valor da diária não pode ser negativo!', 'error')
            return redirect(url_for('diaristas'))

        # Verificar se CPF já existe (se fornecido)
        if cpf:
            diarista_existente = filtrar_por_usuario(
                Diarista.query.filter_by(cpf=cpf), Diarista).first()
            if diarista_existente:
                flash('CPF já está cadastrado para outro diarista!', 'error')
                return redirect(url_for('diaristas'))

        diarista = criar_com_usuario(Diarista,
                                     nome_completo=nome_completo,
                                     cpf=cpf,
                                     telefone=telefone,
                                     valor_diaria=valor_diaria)
        db.session.add(diarista)
        db.session.commit()

        flash('Diarista adicionado com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Valor da diária deve ser um número válido.', 'error')
    except Exception as e:
        flash('Erro ao adicionar diarista. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar diarista: {str(e)}")

    return redirect(url_for('diaristas'))


@app.route('/diaristas/editar/<int:id>', methods=['POST'])
@login_required
def editar_diarista(id):
    try:
        # CORREÇÃO CRÍTICA DE SEGURANÇA: Usar validação user-scoped
        diarista = get_or_404_user_scoped(Diarista, id)

        nome_completo = request.form.get('nome_completo')
        cpf = request.form.get('cpf', '')
        valor_diaria = float(request.form.get('valor_diaria', 0))

        if not nome_completo:
            flash('Nome completo é obrigatório!', 'error')
            return redirect(url_for('diaristas'))

        if valor_diaria < 0:
            flash('Valor da diária não pode ser negativo!', 'error')
            return redirect(url_for('diaristas'))

        # Verificar se CPF já existe para outro diarista (se fornecido)
        if cpf:
            diarista_existente = filtrar_por_usuario(
                Diarista.query.filter_by(cpf=cpf).filter(Diarista.id != id),
                Diarista).first()
            if diarista_existente:
                flash('CPF já está cadastrado para outro diarista!', 'error')
                return redirect(url_for('diaristas'))

        diarista.nome_completo = nome_completo
        diarista.cpf = cpf
        diarista.telefone = request.form.get('telefone', '')
        diarista.valor_diaria = valor_diaria

        db.session.commit()
        flash('Diarista atualizado com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Valor da diária deve ser um número válido.', 'error')
    except Exception as e:
        flash('Erro ao atualizar diarista. Tente novamente.', 'error')
        app.logger.error(f"Erro ao editar diarista: {str(e)}")

    return redirect(url_for('diaristas'))


@app.route('/diaristas/excluir/<int:id>', methods=['POST'])
@login_required
def excluir_diarista(id):
    diarista = obter_por_id_usuario(Diarista, id)
    if not diarista:
        flash('Diarista não encontrado ou acesso negado!', 'error')
        return redirect(url_for('diaristas'))
    
    db.session.delete(diarista)
    db.session.commit()
    flash('Diarista excluído com sucesso!', 'success')
    return redirect(url_for('diaristas'))


# ROTAS DE REGISTROS DE DIARISTAS E RECIBOS
@app.route('/diaristas/registrar-dia', methods=['POST'])
@login_required
def registrar_dia_diarista():
    """Registra um dia de trabalho para um diarista"""
    try:
        diarista_id = int(request.form.get('diarista_id'))
        data = datetime.strptime(request.form.get('data'), '%Y-%m-%d').date()
        hora_entrada = request.form.get('horario_entrada') or None
        hora_saida = request.form.get('horario_saida') or None
        descricao_trabalho = request.form.get('descricao_trabalho', '')
        observacoes = request.form.get('observacoes', '')

        # Verificar se já existe registro para esta data
        registro_existente = filtrar_por_usuario(
            RegistroDiaria.query.filter_by(diarista_id=diarista_id, data=data),
            RegistroDiaria).first()

        if registro_existente:
            return jsonify({
                'success': False,
                'message': 'Já existe um registro para esta data'
            })

        # Converter horários para Time se fornecidos
        if hora_entrada:
            hora_entrada = datetime.strptime(hora_entrada, '%H:%M').time()
        if hora_saida:
            hora_saida = datetime.strptime(hora_saida, '%H:%M').time()

        registro = criar_com_usuario(RegistroDiaria,
                                     diarista_id=diarista_id,
                                     data=data,
                                     hora_entrada=hora_entrada,
                                     hora_saida=hora_saida,
                                     descricao_trabalho=descricao_trabalho,
                                     observacoes=observacoes)

        db.session.add(registro)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Dia registrado com sucesso'
        })

    except Exception as e:
        app.logger.error(f"Erro ao registrar dia: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Erro interno: {str(e)}'
        })


@app.route('/diaristas/<int:diarista_id>/registros')
@login_required
def listar_registros_diarista(diarista_id):
    """Lista os registros de trabalho de um diarista com valores e totais"""
    try:
        diarista = obter_por_id_usuario(Diarista, diarista_id)
        if not diarista:
            return jsonify({'registros': [], 'total_geral': 0, 'quantidade_dias': 0})
        
        registros = filtrar_por_usuario(
            RegistroDiaria.query.filter_by(diarista_id=diarista_id),
            RegistroDiaria).order_by(RegistroDiaria.data.desc()).all()

        total_geral = 0
        registros_json = []
        for registro in registros:
            valor_diaria = diarista.valor_diaria or 0
            total_geral += valor_diaria
            
            registros_json.append({
                'id': registro.id,
                'data': registro.data.isoformat(),
                'horario_entrada': registro.hora_entrada.strftime('%H:%M') if registro.hora_entrada else None,
                'horario_saida': registro.hora_saida.strftime('%H:%M') if registro.hora_saida else None,
                'descricao_trabalho': registro.descricao_trabalho,
                'observacoes': registro.observacoes,
                'valor_diaria': valor_diaria,
                'total_acumulado': total_geral
            })

        # Adicionar informações de resumo
        response_data = {
            'registros': registros_json,
            'total_geral': total_geral,
            'quantidade_dias': len(registros),
            'valor_diaria_atual': diarista.valor_diaria or 0
        }

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Erro ao listar registros: {str(e)}")
        return jsonify({'registros': [], 'total_geral': 0, 'quantidade_dias': 0})


@app.route('/diaristas/<int:diarista_id>/recibo-preview')
@login_required
def preview_recibo_diarista(diarista_id):
    """Gera preview do recibo do diarista"""
    try:
        diarista = obter_por_id_usuario(Diarista, diarista_id)
        if not diarista:
            return '<div class="alert alert-danger">Diarista não encontrado ou acesso negado</div>'
        
        periodo = request.args.get('periodo', 'mes')

        if periodo == 'mes':
            mes_ano = request.args.get('mes_ano')
            if not mes_ano:
                return '<div class="alert alert-danger">Mês/ano é obrigatório</div>'

            ano, mes = map(int, mes_ano.split('-'))

            # Buscar registros do mês
            registros = filtrar_por_usuario(
                RegistroDiaria.query.filter(
                    RegistroDiaria.diarista_id == diarista_id,
                    db.extract('year', RegistroDiaria.data) == ano,
                    db.extract('month', RegistroDiaria.data) == mes),
                RegistroDiaria).order_by(RegistroDiaria.data).all()

        else:  # periodo == 'dias'
            dias_ids = request.args.get('dias', '').split(',')
            if not dias_ids or dias_ids == ['']:
                return '<div class="alert alert-danger">Nenhum dia selecionado</div>'

            registros = filtrar_por_usuario(
                RegistroDiaria.query.filter(RegistroDiaria.id.in_(dias_ids)),
                RegistroDiaria).order_by(RegistroDiaria.data).all()

        if not registros:
            return '<div class="alert alert-warning">Nenhum registro encontrado para o período selecionado</div>'

        # Calcular totais
        total_dias = len(registros)
        valor_total = total_dias * diarista.valor_diaria

        # Gerar HTML do preview
        html = f'''
        <div class="card">
            <div class="card-header bg-success text-white">
                <h5 class="card-title mb-0">Preview do Recibo</h5>
            </div>
            <div class="card-body">
                <div class="row mb-3">
                    <div class="col-md-6">
                        <strong>Diarista:</strong> {diarista.nome_completo}<br>
                        <strong>CPF:</strong> {diarista.cpf or 'Não informado'}<br>
                        <strong>Telefone:</strong> {diarista.telefone or 'Não informado'}
                    </div>
                    <div class="col-md-6 text-end">
                        <strong>Período:</strong> {registros[0].data.strftime('%d/%m/%Y')} a {registros[-1].data.strftime('%d/%m/%Y')}<br>
                        <strong>Total de dias:</strong> {total_dias}<br>
                        <strong>Valor por dia:</strong> R$ {diarista.valor_diaria:.2f}
                    </div>
                </div>

                <div class="table-responsive">
                    <table class="table table-sm table-striped">
                        <thead>
                            <tr>
                                <th>Data</th>
                                <th>Entrada</th>
                                <th>Saída</th>
                                <th>Trabalho Realizado</th>
                                <th>Valor</th>
                            </tr>
                        </thead>
                        <tbody>
        '''

        for registro in registros:
            entrada = registro.hora_entrada.strftime(
                '%H:%M') if registro.hora_entrada else '-'
            saida = registro.hora_saida.strftime(
                '%H:%M') if registro.hora_saida else '-'
            html += f'''
                            <tr>
                                <td>{registro.data.strftime('%d/%m/%Y')}</td>
                                <td>{entrada}</td>
                                <td>{saida}</td>
                                <td>{registro.descricao_trabalho or 'Trabalho geral'}</td>
                                <td>R$ {diarista.valor_diaria:.2f}</td>
                            </tr>
            '''

        html += f'''
                        </tbody>
                        <tfoot>
                            <tr class="table-success">
                                <th colspan="4">TOTAL GERAL</th>
                                <th>R$ {valor_total:.2f}</th>
                            </tr>
                        </tfoot>
                    </table>
                </div>
            </div>
        </div>
        '''

        return html

    except Exception as e:
        app.logger.error(f"Erro ao gerar preview: {str(e)}")
        return f'<div class="alert alert-danger">Erro ao gerar preview: {str(e)}</div>'


@app.route('/diaristas/<int:diarista_id>/recibo')
@login_required
def gerar_recibo_diarista(diarista_id):
    """Gera recibo completo do diarista para impressão/PDF"""
    try:
        diarista = obter_por_id_usuario(Diarista, diarista_id)
        if not diarista:
            flash('Diarista não encontrado ou acesso negado!', 'error')
            return redirect(url_for('diaristas'))
        
        periodo = request.args.get('periodo', 'mes')

        if periodo == 'mes':
            mes_ano = request.args.get('mes_ano')
            if not mes_ano:
                flash('Mês/ano é obrigatório', 'error')
                return redirect(url_for('diaristas'))

            ano, mes = map(int, mes_ano.split('-'))

            registros = RegistroDiaria.query.filter(
                RegistroDiaria.diarista_id == diarista_id,
                db.extract('year', RegistroDiaria.data) == ano,
                db.extract('month', RegistroDiaria.data) == mes).order_by(
                    RegistroDiaria.data).all()

        else:  # periodo == 'dias'
            dias_ids = request.args.get('dias', '').split(',')
            if not dias_ids or dias_ids == ['']:
                flash('Nenhum dia selecionado', 'error')
                return redirect(url_for('diaristas'))

            registros = filtrar_por_usuario(
                RegistroDiaria.query.filter(RegistroDiaria.id.in_(dias_ids)),
                RegistroDiaria).order_by(RegistroDiaria.data).all()

        if not registros:
            flash('Nenhum registro encontrado para o período selecionado',
                  'error')
            return redirect(url_for('diaristas'))

        # Calcular totais
        total_dias = len(registros)
        valor_total = total_dias * diarista.valor_diaria

        return render_template('recibo_diarista.html',
                               diarista=diarista,
                               registros=registros,
                               total_dias=total_dias,
                               valor_total=valor_total,
                               data_emissao=datetime.now())

    except Exception as e:
        app.logger.error(f"Erro ao gerar recibo: {str(e)}")
        flash(f'Erro ao gerar recibo: {str(e)}', 'error')
        return redirect(url_for('diaristas'))


@app.route('/registro-diaria/<int:registro_id>/recibo')
@login_required
def recibo_individual_diarista(registro_id):
    """Gera recibo individual para um dia específico"""
    try:
        registro = obter_por_id_usuario(RegistroDiaria, registro_id)
        if not registro:
            flash('Registro não encontrado ou acesso negado!', 'error')
            return redirect(url_for('diaristas'))
        
        diarista = registro.diarista

        return render_template('recibo_individual.html',
                               diarista=diarista,
                               registro=registro,
                               data_emissao=datetime.now())

    except Exception as e:
        app.logger.error(f"Erro ao gerar recibo individual: {str(e)}")
        flash(f'Erro ao gerar recibo: {str(e)}', 'error')
        return redirect(url_for('diaristas'))


# ROTAS DE FORNECEDORES
@app.route('/fornecedores')
@login_required
def fornecedores():
    """Página principal de gestão de fornecedores"""
    fornecedores = filtrar_por_usuario(
        Fornecedor.query.order_by(Fornecedor.nome),
        Fornecedor
    ).all()

    # Estatísticas
    total_ativos = sum(1 for f in fornecedores if f.status == 'ativo')
    total_inativos = sum(1 for f in fornecedores if f.status == 'inativo')

    return render_template('fornecedores.html',
                         fornecedores=fornecedores,
                         total_ativos=total_ativos,
                         total_inativos=total_inativos)


@app.route('/fornecedores/adicionar', methods=['POST'])
@login_required
def adicionar_fornecedor():
    try:
        nome = request.form.get('nome', '').strip()
        nome_fantasia = request.form.get('nome_fantasia', '').strip()
        cnpj_cpf = request.form.get('cnpj_cpf', '').strip()
        categoria = request.form.get('categoria', '').strip()
        contato_nome = request.form.get('contato_nome', '').strip()
        telefone = request.form.get('telefone', '').strip()
        email = request.form.get('email', '').strip()
        endereco = request.form.get('endereco', '').strip()
        cidade = request.form.get('cidade', '').strip()
        estado = request.form.get('estado', '').strip()
        cep = request.form.get('cep', '').strip()
        observacoes = request.form.get('observacoes', '').strip()

        if not nome or not categoria:
            flash('Nome e categoria são obrigatórios!', 'error')
            return redirect(url_for('fornecedores'))

        fornecedor = criar_com_usuario(
            Fornecedor,
            nome=nome,
            nome_fantasia=nome_fantasia if nome_fantasia else None,
            cnpj_cpf=cnpj_cpf if cnpj_cpf else None,
            categoria=categoria,
            contato_nome=contato_nome if contato_nome else None,
            telefone=telefone if telefone else None,
            email=email if email else None,
            endereco=endereco if endereco else None,
            cidade=cidade if cidade else None,
            estado=estado if estado else None,
            cep=cep if cep else None,
            observacoes=observacoes if observacoes else None,
            status='ativo'
        )

        db.session.add(fornecedor)
        db.session.commit()

        flash('Fornecedor adicionado com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao adicionar fornecedor. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar fornecedor: {str(e)}")

    return redirect(url_for('fornecedores'))


@app.route('/fornecedores/editar/<int:id>', methods=['POST'])
@login_required
def editar_fornecedor(id):
    try:
        fornecedor = get_or_404_user_scoped(Fornecedor, id)

        fornecedor.nome = request.form.get('nome', '').strip()
        fornecedor.nome_fantasia = request.form.get('nome_fantasia', '').strip() or None
        fornecedor.cnpj_cpf = request.form.get('cnpj_cpf', '').strip() or None
        fornecedor.categoria = request.form.get('categoria', '').strip()
        fornecedor.contato_nome = request.form.get('contato_nome', '').strip() or None
        fornecedor.telefone = request.form.get('telefone', '').strip() or None
        fornecedor.email = request.form.get('email', '').strip() or None
        fornecedor.endereco = request.form.get('endereco', '').strip() or None
        fornecedor.cidade = request.form.get('cidade', '').strip() or None
        fornecedor.estado = request.form.get('estado', '').strip() or None
        fornecedor.cep = request.form.get('cep', '').strip() or None
        fornecedor.observacoes = request.form.get('observacoes', '').strip() or None

        if not fornecedor.nome or not fornecedor.categoria:
            flash('Nome e categoria são obrigatórios!', 'error')
            return redirect(url_for('fornecedores'))

        db.session.commit()
        flash('Fornecedor atualizado com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao atualizar fornecedor. Tente novamente.', 'error')
        app.logger.error(f"Erro ao editar fornecedor: {str(e)}")

    return redirect(url_for('fornecedores'))


@app.route('/fornecedores/excluir/<int:id>', methods=['POST'])
@login_required
def excluir_fornecedor(id):
    try:
        fornecedor = obter_por_id_usuario(Fornecedor, id)
        if not fornecedor:
            flash('Fornecedor não encontrado ou acesso negado!', 'error')
            return redirect(url_for('fornecedores'))

        db.session.delete(fornecedor)
        db.session.commit()
        flash('Fornecedor excluído com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao excluir fornecedor. Tente novamente.', 'error')
        app.logger.error(f"Erro ao excluir fornecedor: {str(e)}")

    return redirect(url_for('fornecedores'))


@app.route('/fornecedores/ativar/<int:id>', methods=['POST'])
@login_required
def ativar_fornecedor(id):
    try:
        fornecedor = get_or_404_user_scoped(Fornecedor, id)
        fornecedor.status = 'ativo'
        db.session.commit()
        flash('Fornecedor ativado com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao ativar fornecedor. Tente novamente.', 'error')
        app.logger.error(f"Erro ao ativar fornecedor: {str(e)}")

    return redirect(url_for('fornecedores'))


@app.route('/fornecedores/desativar/<int:id>', methods=['POST'])
@login_required
def desativar_fornecedor(id):
    try:
        fornecedor = get_or_404_user_scoped(Fornecedor, id)
        fornecedor.status = 'inativo'
        db.session.commit()
        flash('Fornecedor desativado com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao desativar fornecedor. Tente novamente.', 'error')
        app.logger.error(f"Erro ao desativar fornecedor: {str(e)}")

    return redirect(url_for('fornecedores'))


# ROTAS DE CONTAS A PAGAR (MÓDULO FINANCEIRO)
@app.route('/contas-pagar')
@login_required
def contas_pagar():
    """Página principal de contas a pagar"""
    contas = filtrar_por_usuario(
        ContasPagar.query.order_by(ContasPagar.data_vencimento.desc()),
        ContasPagar
    ).all()
    
    # Estatísticas
    total_pendente = sum(c.valor for c in contas if c.status == 'pendente')
    total_vencido = sum(c.valor for c in contas if c.esta_vencida())
    total_pago = sum(c.valor for c in contas if c.status == 'pago')
    
    return render_template('contas_pagar.html',
                         contas=contas,
                         total_pendente=total_pendente,
                         total_vencido=total_vencido,
                         total_pago=total_pago,
                         datetime=datetime)


@app.route('/contas-pagar/adicionar', methods=['POST'])
@login_required
def adicionar_conta_pagar():
    try:
        descricao = request.form.get('descricao', '').strip()
        fornecedor = request.form.get('fornecedor', '').strip()
        categoria = request.form.get('categoria', '').strip()
        valor = float(request.form.get('valor', 0))
        data_vencimento_str = request.form.get('data_vencimento')
        numero_documento = request.form.get('numero_documento', '').strip()
        observacoes = request.form.get('observacoes', '').strip()
        
        if not descricao or not fornecedor or not categoria:
            flash('Descrição, fornecedor e categoria são obrigatórios!', 'error')
            return redirect(url_for('contas_pagar'))
        
        if valor <= 0:
            flash('Valor deve ser maior que zero!', 'error')
            return redirect(url_for('contas_pagar'))
        
        if not data_vencimento_str:
            flash('Data de vencimento é obrigatória!', 'error')
            return redirect(url_for('contas_pagar'))
        
        data_vencimento = datetime.strptime(data_vencimento_str, '%Y-%m-%d').date()
        
        conta = criar_com_usuario(
            ContasPagar,
            descricao=descricao,
            fornecedor=fornecedor,
            categoria=categoria,
            valor=valor,
            data_vencimento=data_vencimento,
            numero_documento=numero_documento if numero_documento else None,
            observacoes=observacoes if observacoes else None
        )
        
        db.session.add(conta)
        db.session.commit()
        
        flash('Conta a pagar adicionada com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Verifique os valores fornecidos.', 'error')
        app.logger.error(f"Erro ao adicionar conta a pagar: {str(e)}")
    except Exception as e:
        flash('Erro ao adicionar conta. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar conta a pagar: {str(e)}")
    
    return redirect(url_for('contas_pagar'))


@app.route('/contas-pagar/editar/<int:id>', methods=['POST'])
@login_required
def editar_conta_pagar(id):
    try:
        conta = get_or_404_user_scoped(ContasPagar, id)
        
        descricao = request.form.get('descricao', '').strip()
        fornecedor = request.form.get('fornecedor', '').strip()
        categoria = request.form.get('categoria', '').strip()
        valor = float(request.form.get('valor', 0))
        data_vencimento_str = request.form.get('data_vencimento')
        numero_documento = request.form.get('numero_documento', '').strip()
        observacoes = request.form.get('observacoes', '').strip()
        
        if not descricao or not fornecedor or not categoria:
            flash('Descrição, fornecedor e categoria são obrigatórios!', 'error')
            return redirect(url_for('contas_pagar'))
        
        if valor <= 0:
            flash('Valor deve ser maior que zero!', 'error')
            return redirect(url_for('contas_pagar'))
        
        conta.descricao = descricao
        conta.fornecedor = fornecedor
        conta.categoria = categoria
        conta.valor = valor
        conta.numero_documento = numero_documento if numero_documento else None
        conta.observacoes = observacoes if observacoes else None
        
        if data_vencimento_str:
            conta.data_vencimento = datetime.strptime(data_vencimento_str, '%Y-%m-%d').date()
        
        db.session.commit()
        flash('Conta atualizada com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Verifique os valores fornecidos.', 'error')
        app.logger.error(f"Erro ao editar conta: {str(e)}")
    except Exception as e:
        flash('Erro ao atualizar conta. Tente novamente.', 'error')
        app.logger.error(f"Erro ao editar conta: {str(e)}")
    
    return redirect(url_for('contas_pagar'))


@app.route('/contas-pagar/excluir/<int:id>', methods=['POST'])
@login_required
def excluir_conta_pagar(id):
    try:
        conta = obter_por_id_usuario(ContasPagar, id)
        if not conta:
            flash('Conta não encontrada ou acesso negado!', 'error')
            return redirect(url_for('contas_pagar'))
        
        db.session.delete(conta)
        db.session.commit()
        flash('Conta excluída com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao excluir conta. Tente novamente.', 'error')
        app.logger.error(f"Erro ao excluir conta: {str(e)}")
    
    return redirect(url_for('contas_pagar'))


@app.route('/contas-pagar/pagar/<int:id>', methods=['POST'])
@login_required
def pagar_conta(id):
    try:
        conta = get_or_404_user_scoped(ContasPagar, id)
        
        if conta.status == 'pago':
            flash('Esta conta já foi paga!', 'warning')
            return redirect(url_for('contas_pagar'))
        
        forma_pagamento = request.form.get('forma_pagamento', '').strip()
        data_pagamento_str = request.form.get('data_pagamento')
        
        if not forma_pagamento:
            flash('Forma de pagamento é obrigatória!', 'error')
            return redirect(url_for('contas_pagar'))
        
        conta.status = 'pago'
        conta.forma_pagamento = forma_pagamento
        
        if data_pagamento_str:
            conta.data_pagamento = datetime.strptime(data_pagamento_str, '%Y-%m-%d').date()
        else:
            conta.data_pagamento = datetime.now().date()
        
        db.session.commit()
        flash('Pagamento registrado com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao registrar pagamento. Tente novamente.', 'error')
        app.logger.error(f"Erro ao pagar conta: {str(e)}")
    
    return redirect(url_for('contas_pagar'))


@app.route('/contas-pagar/cancelar/<int:id>', methods=['POST'])
@login_required
def cancelar_conta(id):
    try:
        conta = get_or_404_user_scoped(ContasPagar, id)
        
        if conta.status == 'pago':
            flash('Não é possível cancelar uma conta já paga!', 'error')
            return redirect(url_for('contas_pagar'))
        
        conta.status = 'cancelado'
        db.session.commit()
        flash('Conta cancelada com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao cancelar conta. Tente novamente.', 'error')
        app.logger.error(f"Erro ao cancelar conta: {str(e)}")
    
    return redirect(url_for('contas_pagar'))


# ROTAS DE SILOS
@app.route('/silos')
@login_required
def silos():
    """Página de silos otimizada para performance"""
    # Query otimizada com isolamento de usuário - buscar apenas dados necessários
    silos_list = filtrar_por_usuario(
        Silo.query.options(
            db.joinedload(Silo.movimentacoes).joinedload(
                MovimentacaoSilo.grao)), Silo).all()

    graos_list = filtrar_por_usuario(Grao.query, Grao).all()

    # Buscar talhões para seleção no formulário
    talhoes_list = filtrar_por_usuario(
        Talhao.query.order_by(Talhao.nome), Talhao
    ).all()

    return render_template('silos.html', silos=silos_list, graos=graos_list, talhoes=talhoes_list)


@app.route('/silos/relatorio-csv')
@login_required
def relatorio_silos_csv():
    """Gera relatório CSV de entrada de grãos nos silos"""
    try:
        # Buscar todas as movimentações de ENTRADA do usuário, ordenadas por data
        movimentacoes = filtrar_por_usuario(
            MovimentacaoSilo.query
            .filter_by(tipo_movimentacao='Entrada')
            .order_by(MovimentacaoSilo.data_movimentacao.desc()),
            MovimentacaoSilo
        ).all()

        # Criar buffer de string para o CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Cabeçalho do CSV baseado no modelo fornecido
        writer.writerow([
            'DATA',
            'HORA',
            'MOTORISTA',
            'SILO',
            'TALHÃO',
            'PLACA CAMINHÃO',
            'PRODUTO',
            'PESO INICIAL CARREGADO',
            'TARA',
            'UMIDADE',
            'PESO BRUTO',
            'PESO LÍQUIDO'
        ])

        # Preencher dados
        for mov in movimentacoes:
            writer.writerow([
                mov.data_movimentacao.strftime('%d/%m/%Y'),
                mov.data_movimentacao.strftime('%H:%M'),
                mov.nome_motorista or '',
                mov.silo.nome if mov.silo else '',
                mov.talhao or '',  # Usar campo talhao diretamente
                mov.placa_caminhao or '',
                mov.grao.nome if mov.grao else '',
                f'{mov.peso_entrada_kg:.2f}' if mov.peso_entrada_kg else '',
                f'{mov.peso_saida_kg:.2f}' if mov.peso_saida_kg else '',
                f'{mov.umidade:.1f}%' if mov.umidade else '',  # Usar campo umidade diretamente
                f'{mov.peso_entrada_kg:.2f}' if mov.peso_entrada_kg else '',  # PESO BRUTO = PESO INICIAL
                f'{mov.peso_liquido_kg:.2f}' if mov.peso_liquido_kg else f'{mov.quantidade_kg:.2f}'
            ])

        # Preparar resposta CSV
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8-sig'  # UTF-8 com BOM para Excel
        response.headers['Content-Disposition'] = f'attachment; filename=relatorio_entrada_graos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        return response

    except Exception as e:
        app.logger.error(f"Erro ao gerar relatório CSV: {str(e)}")
        flash('Erro ao gerar relatório. Tente novamente.', 'error')
        return redirect(url_for('silos'))


@app.route('/silos/relatorio-pdf')
@login_required
def relatorio_silos_pdf():
    """Gera relatório PDF de entrada de grãos nos silos com formatação profissional"""
    try:
        # Buscar todas as movimentações de ENTRADA do usuário, ordenadas por data
        movimentacoes = filtrar_por_usuario(
            MovimentacaoSilo.query
            .filter_by(tipo_movimentacao='Entrada')
            .order_by(MovimentacaoSilo.data_movimentacao.desc()),
            MovimentacaoSilo
        ).all()

        # Criar buffer para o PDF
        buffer = io.BytesIO()

        # Definir função para adicionar logo em todas as páginas (flutuante)
        logo_path = os.path.join(os.path.dirname(__file__), '..', 'CropLink PNG.png')

        def add_logo(canvas, doc):
            """Adiciona logo no canto superior esquerdo sem empurrar conteúdo"""
            if os.path.exists(logo_path):
                # Logo com 160mm de largura (metade de 320mm)
                # Posição: -10mm da esquerda (além da borda), 50mm abaixo do topo
                canvas.drawImage(logo_path, -10*mm, landscape(A4)[1] - 50*mm,
                                width=160*mm, height=50*mm,
                                preserveAspectRatio=True, mask='auto')

        # Criar documento PDF em modo paisagem (landscape) para caber todas as colunas
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(A4),
            rightMargin=10*mm,
            leftMargin=10*mm,
            topMargin=15*mm,
            bottomMargin=15*mm
        )

        # Container para os elementos do PDF
        elements = []

        # Estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c5f2d'),
            spaceAfter=12,
            alignment=1  # Centralizado
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=20,
            alignment=1
        )

        # Título
        title = Paragraph("CONTROLE DE ENTRADA DE GRÃOS", title_style)
        elements.append(title)

        # Subtítulo com data de geração
        subtitle = Paragraph(
            f"Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}",
            subtitle_style
        )
        elements.append(subtitle)

        # Preparar dados da tabela
        data = [[
            'DATA',
            'HORA',
            'MOTORISTA',
            'SILO',
            'TALHÃO',
            'PLACA\nCAMINHÃO',
            'PRODUTO',
            'PESO INICIAL\nCARREGADO (kg)',
            'TARA\n(kg)',
            'UMIDADE',
            'PESO\nBRUTO (kg)',
            'PESO\nLÍQUIDO (kg)'
        ]]

        # Preencher dados e calcular total
        total_peso_liquido = 0

        for mov in movimentacoes:
            # Calcular peso líquido para o total
            peso_liquido = mov.peso_liquido_kg or mov.quantidade_kg or 0
            total_peso_liquido += peso_liquido

            # Obter nome do talhão (prioriza relacionamento, fallback para campo texto)
            talhao_nome = ''
            try:
                if mov.talhao_origem:
                    talhao_nome = mov.talhao_origem.nome[:15]
                elif mov.talhao:
                    talhao_nome = mov.talhao[:15]
            except:
                talhao_nome = ''

            # Obter hora da movimentação (extrair do campo data_movimentacao que é DateTime)
            try:
                hora_str = mov.data_movimentacao.strftime('%H:%M') if mov.data_movimentacao else ''
            except:
                hora_str = ''

            data.append([
                mov.data_movimentacao.strftime('%d/%m/%Y') if mov.data_movimentacao else '',
                hora_str,
                (mov.nome_motorista or '')[:15],  # Limitar nome
                (mov.silo.nome if mov.silo else '')[:10],
                talhao_nome,
                mov.placa_caminhao or '',
                (mov.grao.nome if mov.grao else '')[:10],
                f'{mov.peso_entrada_kg:,.0f}'.replace(',', '.') if mov.peso_entrada_kg else '',
                f'{mov.peso_saida_kg:,.0f}'.replace(',', '.') if mov.peso_saida_kg else '',
                f'{mov.umidade:.1f}%' if mov.umidade else '',
                f'{mov.peso_entrada_kg:,.0f}'.replace(',', '.') if mov.peso_entrada_kg else '',
                f'{peso_liquido:,.0f}'.replace(',', '.')
            ])

        # Adicionar linha de TOTAL (somente se houver movimentações)
        if movimentacoes:
            data.append([
                '', '', '', '', '', '', '',
                '', '', '',
                'TOTAL:',
                f'{total_peso_liquido:,.0f}'.replace(',', '.')
            ])

        # Criar tabela
        table = Table(data, repeatRows=1)

        # Estilo da tabela - profissional e bonito
        # Determinar estilos base
        base_styles = [
            # Cabeçalho
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5f2d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),

            # Bordas
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#2c5f2d')),

            # Alinhamento especial para colunas numéricas
            ('ALIGN', (7, 1), (-1, -1), 'RIGHT'),
        ]

        # Estilos para os dados
        if movimentacoes:
            # Se tem movimentações, aplicar zebrado exceto na última linha (TOTAL)
            base_styles.extend([
                ('BACKGROUND', (0, 1), (-1, -2), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -2), colors.black),
                ('ALIGN', (0, 1), (-1, -2), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -2), 7),
                ('TOPPADDING', (0, 1), (-1, -2), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -2), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#f0f0f0')]),

                # Linha de TOTAL (última linha)
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#2c5f2d')),
                ('TEXTCOLOR', (0, -1), (-1, -1), colors.whitesmoke),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, -1), (-1, -1), 9),
                ('ALIGN', (10, -1), (-1, -1), 'RIGHT'),
                ('TOPPADDING', (0, -1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, -1), (-1, -1), 8),
                ('LINEABOVE', (0, -1), (-1, -1), 2, colors.HexColor('#2c5f2d')),
            ])
        else:
            # Se não tem movimentações, estilo simples
            base_styles.extend([
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
            ])

        table.setStyle(TableStyle(base_styles))

        elements.append(table)

        # Adicionar rodapé com total de registros
        elements.append(Spacer(1, 10*mm))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=2  # Direita
        )
        footer = Paragraph(
            f"Total de registros: {len(movimentacoes)}",
            footer_style
        )
        elements.append(footer)

        # Gerar PDF com logo flutuante
        doc.build(elements, onFirstPage=add_logo, onLaterPages=add_logo)

        # Preparar resposta
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=relatorio_entrada_graos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

        return response

    except Exception as e:
        app.logger.error(f"Erro ao gerar relatório PDF: {str(e)}")
        flash('Erro ao gerar relatório PDF. Tente novamente.', 'error')
        return redirect(url_for('silos'))


@app.route('/silos/adicionar', methods=['POST'])
@login_required
def adicionar_silo():
    try:
        nome = request.form.get('nome', '').strip()
        capacidade_kg = float(request.form.get('capacidade_kg', 0))

        if not nome:
            flash('Nome do silo é obrigatório!', 'error')
            return redirect(url_for('silos'))

        if capacidade_kg <= 0:
            flash('Capacidade deve ser maior que zero!', 'error')
            return redirect(url_for('silos'))

        # Verificar se já existe silo com mesmo nome (case-insensitive)
        silo_existente = filtrar_por_usuario(
            Silo.query.filter(func.upper(Silo.nome) == func.upper(nome)),
            Silo).first()
        if silo_existente:
            flash(f'Já existe um silo com o nome "{nome}"! Use outro nome.',
                  'error')
            return redirect(url_for('silos'))

        silo = criar_com_usuario(Silo, nome=nome, capacidade_kg=capacidade_kg)
        db.session.add(silo)
        db.session.commit()

        flash(f'Silo "{nome}" adicionado com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Capacidade deve ser um número válido.', 'error')
    except Exception as e:
        db.session.rollback()
        flash('Erro ao adicionar silo. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar silo: {str(e)}")

    return redirect(url_for('silos'))


@app.route('/silos/<int:silo_id>/deletar', methods=['POST'])
@login_required
def deletar_silo(silo_id):
    """Deleta um silo se não tiver movimentações"""
    try:
        silo = obter_por_id_usuario(Silo, silo_id)
        if not silo:
            flash('Silo não encontrado ou acesso negado!', 'error')
            return redirect(url_for('silos'))

        # Verificar se há movimentações no silo
        movimentacoes = filtrar_por_usuario(
            MovimentacaoSilo.query.filter_by(silo_id=silo_id),
            MovimentacaoSilo).count()
        if movimentacoes > 0:
            flash(
                f'Não é possível deletar o silo "{silo.nome}" porque ele possui movimentações registradas!',
                'error')
            return redirect(url_for('silos'))

        nome_silo = silo.nome
        db.session.delete(silo)
        db.session.commit()

        flash(f'Silo "{nome_silo}" deletado com sucesso!', 'success')

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erro ao deletar silo: {str(e)}")
        flash('Erro ao deletar silo. Tente novamente.', 'error')

    return redirect(url_for('silos'))


@app.route('/silos/<int:silo_id>/esvaziar', methods=['POST'])
@login_required
def esvaziar_silo(silo_id):
    """Esvazia completamente um silo - remove todas movimentações e zera quantidades"""
    try:
        silo = obter_por_id_usuario(Silo, silo_id)
        if not silo:
            flash('Silo não encontrado ou acesso negado!', 'error')
            return redirect(url_for('silos'))

        nome_silo = silo.nome
        
        # Contar movimentações antes da remoção
        movimentacoes_count = filtrar_por_usuario(
            MovimentacaoSilo.query.filter_by(silo_id=silo_id),
            MovimentacaoSilo).count()
        
        # Remover TODAS as movimentações deste silo (para este usuário)
        movimentacoes = filtrar_por_usuario(
            MovimentacaoSilo.query.filter_by(silo_id=silo_id),
            MovimentacaoSilo)
        
        for mov in movimentacoes:
            db.session.delete(mov)
        
        # Calcular grãos diferentes antes da remoção (para estatísticas)
        graos_diferentes = db.session.query(MovimentacaoSilo.grao_id)\
            .filter(MovimentacaoSilo.silo_id == silo_id,
                   MovimentacaoSilo.user_id == current_user.id)\
            .distinct().count()
        
        db.session.commit()

        flash(f'Silo "{nome_silo}" esvaziado com sucesso! '
              f'Removidas {movimentacoes_count} movimentações de {graos_diferentes} tipos de grãos diferentes. '
              f'Agora é possível deletar o silo.', 'success')

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erro ao esvaziar silo: {str(e)}")
        flash('Erro ao esvaziar silo. Tente novamente.', 'error')

    return redirect(url_for('silos'))


@app.route('/silos/editar/<int:id>', methods=['POST'])
@login_required
def editar_silo(id):
    try:
        # SEGURANÇA: Usar função segura de validação
        silo = get_or_404_user_scoped(Silo, id)

        silo.nome = request.form.get('nome')
        capacidade_kg = float(request.form.get('capacidade_kg', 0))

        if not silo.nome:
            flash('Nome do silo é obrigatório!', 'error')
            return redirect(url_for('silos'))

        if capacidade_kg <= 0:
            flash('Capacidade deve ser maior que zero!', 'error')
            return redirect(url_for('silos'))

        silo.capacidade_kg = capacidade_kg
        db.session.commit()

        flash('Silo atualizado com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Capacidade deve ser um número válido.', 'error')
    except Exception as e:
        flash('Erro ao atualizar silo. Tente novamente.', 'error')
        app.logger.error(f"Erro ao editar silo: {str(e)}")

    return redirect(url_for('silos'))




@app.route('/graos/adicionar', methods=['POST'])
@login_required
def adicionar_grao():
    nome = request.form.get('nome')

    grao = criar_com_usuario(Grao, nome=nome)
    db.session.add(grao)
    db.session.commit()

    flash('Grão adicionado com sucesso!', 'success')
    return redirect(url_for('silos'))


# NOVAS ROTAS PARA MOVIMENTAÇÃO DE GRÃOS EM SILOS


@app.route('/silos/movimentacao', methods=['POST'])
@login_required
def movimentacao_silo():
    """Registra movimentação (entrada ou saída) de grãos em silos"""
    try:
        tipo = request.form.get('tipo')  # 'Entrada' ou 'Saída'
        silo_id = int(request.form.get('silo_id'))
        
        # Capturar dados de transporte PRIMEIRO para calcular quantidade
        placa_caminhao = request.form.get('placa_caminhao', '').strip()
        nome_motorista = request.form.get('nome_motorista', '').strip()
        peso_entrada_kg = request.form.get('peso_entrada_kg', '')
        peso_saida_kg = request.form.get('peso_saida_kg', '')
        observacao = request.form.get('observacao', '')

        # Capturar dados de origem e qualidade
        talhao = request.form.get('talhao', '').strip()  # Campo texto livre (deprecated)
        talhao_id_str = request.form.get('talhao_id', '').strip()
        talhao_id = int(talhao_id_str) if talhao_id_str else None
        umidade_str = request.form.get('umidade', '').strip()
        umidade = float(umidade_str) if umidade_str else None

        # Converter pesos (aceita valores sem pontos como 1000000 = 1 milhão)
        def converter_peso(valor_str):
            if not valor_str:
                return None
            # Remove caracteres não numéricos e converte
            numero_limpo = ''.join(filter(str.isdigit, str(valor_str)))
            return float(numero_limpo) if numero_limpo else None

        peso_entrada_convertido = converter_peso(peso_entrada_kg)
        peso_saida_convertido = converter_peso(peso_saida_kg)

        # CALCULAR quantidade automaticamente DOS PESOS ou pegar manual
        quantidade_kg = 0
        peso_liquido_calculado = None

        if peso_entrada_convertido and peso_saida_convertido:
            # Cálculo automático baseado no tipo de movimentação
            if tipo == 'Entrada':
                # ENTRADA NO SILO:
                # Peso Entrada = Caminhão CHEIO
                # Peso Saída = Caminhão VAZIO
                # Peso Líquido = Entrada - Saída
                peso_liquido_calculado = peso_entrada_convertido - peso_saida_convertido

                if peso_liquido_calculado < 0:
                    flash('Erro: Para entrada, o peso de entrada (caminhão cheio) deve ser maior que o peso de saída (caminhão vazio)!', 'error')
                    return redirect(url_for('silos'))
            else:  # tipo == 'Saída'
                # SAÍDA DO SILO:
                # Peso Entrada = Caminhão VAZIO
                # Peso Saída = Caminhão CHEIO
                # Peso Líquido = Saída - Entrada
                peso_liquido_calculado = peso_saida_convertido - peso_entrada_convertido

                if peso_liquido_calculado < 0:
                    flash('Erro: Para saída, o peso de saída (caminhão cheio) deve ser maior que o peso de entrada (caminhão vazio)!', 'error')
                    return redirect(url_for('silos'))

            quantidade_kg = peso_liquido_calculado
            app.logger.info(f"Quantidade calculada automaticamente ({tipo}): {quantidade_kg} kg (Entrada: {peso_entrada_convertido}, Saída: {peso_saida_convertido})")
        else:
            # Tentar pegar quantidade manual se não tem pesos
            quantidade_str = request.form.get('quantidade_kg', '').strip()
            if quantidade_str:
                try:
                    quantidade_kg = float(quantidade_str)
                except ValueError:
                    flash('Erro: Valores numéricos inválidos.', 'error')
                    return redirect(url_for('silos'))
            else:
                flash('É obrigatório informar os pesos de entrada e saída do caminhão OU uma quantidade manual!', 'error')
                return redirect(url_for('silos'))

        # Validações
        if not tipo or tipo not in ['Entrada', 'Saída']:
            flash('Tipo de movimentação inválido!', 'error')
            return redirect(url_for('silos'))

        if quantidade_kg <= 0:
            flash('Quantidade deve ser maior que zero!', 'error')
            return redirect(url_for('silos'))

        silo = obter_por_id_usuario(Silo, silo_id)
        if not silo:
            flash('Silo não encontrado ou acesso negado!', 'error')
            return redirect(url_for('silos'))

        # Para ENTRADA: usar nome do grão (texto livre) e criar/buscar o grão
        if tipo == 'Entrada':
            nome_grao = request.form.get('nome_grao', '').strip()
            if not nome_grao:
                flash('Nome do grão é obrigatório para entrada!', 'error')
                return redirect(url_for('silos'))

            # Buscar ou criar o grão
            grao = filtrar_por_usuario(Grao.query.filter_by(nome=nome_grao), Grao).first()
            if not grao:
                grao = criar_com_usuario(Grao, nome=nome_grao)
                db.session.add(grao)
                db.session.flush()  # Para obter o ID

            # Verificar capacidade do silo
            capacidade_disponivel = silo.get_capacidade_disponivel()
            if quantidade_kg > capacidade_disponivel:
                flash(
                    f'Capacidade insuficiente! Disponível: {capacidade_disponivel:.1f} kg',
                    'error')
                return redirect(url_for('silos'))

        # Para SAÍDA: usar ID do grão selecionado
        else:  # tipo == 'Saída'
            grao_id = request.form.get('grao_id')
            if not grao_id:
                flash('Selecione o grão para saída!', 'error')
                return redirect(url_for('silos'))

            grao = obter_por_id_usuario(Grao, int(grao_id))
            if not grao:
                flash('Grão não encontrado ou acesso negado!', 'error')
                return redirect(url_for('silos'))

            # Verificar estoque disponível
            estoque_atual = silo.get_estoque_por_grao(grao.id)
            if quantidade_kg > estoque_atual:
                flash(
                    f'Estoque insuficiente! Disponível: {estoque_atual:.1f} kg de {grao.nome}',
                    'error')
                return redirect(url_for('silos'))


        # Criar registro de movimentação
        movimentacao = criar_com_usuario(MovimentacaoSilo,
            tipo_movimentacao=tipo,
            quantidade_kg=quantidade_kg,
            observacao=observacao,
            placa_caminhao=placa_caminhao,
            nome_motorista=nome_motorista,
            peso_entrada_kg=peso_entrada_convertido,
            peso_saida_kg=peso_saida_convertido,
            peso_liquido_kg=peso_liquido_calculado,
            talhao=talhao,
            talhao_id=talhao_id,
            umidade=umidade,
            silo_id=silo_id,
            grao_id=grao.id)

        db.session.add(movimentacao)
        db.session.commit()

        # INVALIDAR CACHE após movimentação de silo
        CacheInvalidationEvents.on_silo_movement(current_user.id, silo_id)

        # Calcular sacas para feedback
        sacas = quantidade_kg / 60.0

        flash(
            f'{tipo} de {quantidade_kg:.1f} kg ({sacas:.1f} sacas) de {grao.nome} registrada com sucesso!',
            'success')

    except ValueError as e:
        flash('Erro: Valores numéricos inválidos.', 'error')
        app.logger.error(f"Erro de valor na movimentação: {str(e)}")
    except Exception as e:
        flash('Erro ao registrar movimentação. Tente novamente.', 'error')
        app.logger.error(f"Erro na movimentação de silo: {str(e)}")

    return redirect(url_for('silos'))


# Nova rota para carregar grãos por silo (AJAX)
@app.route('/api/silos/<int:silo_id>/graos')
@login_required
def get_graos_por_silo(silo_id):
    """Retorna os grãos disponíveis em um silo específico (OTIMIZADO)"""
    try:
        # SQL com filtro de usuário
        if current_user.is_admin:
            estoques_query = text("""
                SELECT 
                    g.id, 
                    g.nome,
                    SUM(CASE WHEN ms.tipo_movimentacao = 'Entrada' THEN ms.quantidade_kg ELSE -ms.quantidade_kg END) as estoque
                FROM movimentacao_silo ms
                JOIN grao g ON ms.grao_id = g.id
                WHERE ms.silo_id = :silo_id
                GROUP BY g.id, g.nome
                HAVING SUM(CASE WHEN ms.tipo_movimentacao = 'Entrada' THEN ms.quantidade_kg ELSE -ms.quantidade_kg END) > 0
                ORDER BY g.nome
            """)
            result = db.session.execute(estoques_query, {
                'silo_id': silo_id
            }).fetchall()
        else:
            estoques_query = text("""
                SELECT 
                    g.id, 
                    g.nome,
                    SUM(CASE WHEN ms.tipo_movimentacao = 'Entrada' THEN ms.quantidade_kg ELSE -ms.quantidade_kg END) as estoque
                FROM movimentacao_silo ms
                JOIN grao g ON ms.grao_id = g.id
                WHERE ms.silo_id = :silo_id AND ms.user_id = :user_id
                GROUP BY g.id, g.nome
                HAVING SUM(CASE WHEN ms.tipo_movimentacao = 'Entrada' THEN ms.quantidade_kg ELSE -ms.quantidade_kg END) > 0
                ORDER BY g.nome
            """)
            result = db.session.execute(estoques_query, {
                'silo_id': silo_id,
                'user_id': current_user.id
            }).fetchall()

        graos_disponiveis = [{
            'id': row[0],
            'nome': row[1],
            'estoque': float(row[2])
        } for row in result]

        return jsonify(graos_disponiveis)

    except Exception as e:
        app.logger.error(f"Erro ao buscar grãos do silo: {str(e)}")
        return jsonify([]), 500


@app.route('/silos/<int:silo_id>/historico')
@login_required
def historico_silo(silo_id):
    """Exibe o histórico de movimentações de um silo específico"""
    silo = obter_por_id_usuario(Silo, silo_id)
    if not silo:
        flash('Silo não encontrado ou acesso negado!', 'error')
        return redirect(url_for('silos'))
    
    movimentacoes = filtrar_por_usuario(
        MovimentacaoSilo.query.filter_by(silo_id=silo_id),
        MovimentacaoSilo).order_by(MovimentacaoSilo.data_movimentacao.desc()).limit(100).all()

    return render_template('historico_silo.html',
                           silo=silo,
                           movimentacoes=movimentacoes)


@app.route('/silos/relatorio')
@login_required
def relatorio_silos():
    """Gera relatório completo dos silos com análises de capacidade e estoque (OTIMIZADO)"""
    try:
        # Consulta para relatório com filtro de usuário
        if current_user.is_admin:
            relatorio_query = text("""
                SELECT 
                    s.id as silo_id,
                    s.nome as silo_nome,
                    s.capacidade_kg,
                    g.id as grao_id,
                    g.nome as grao_nome,
                    COALESCE(SUM(CASE WHEN ms.tipo_movimentacao = 'Entrada' THEN ms.quantidade_kg ELSE -ms.quantidade_kg END), 0) as estoque_grao
                FROM silo s
                LEFT JOIN movimentacao_silo ms ON s.id = ms.silo_id
                LEFT JOIN grao g ON ms.grao_id = g.id
                GROUP BY s.id, s.nome, s.capacidade_kg, g.id, g.nome
                ORDER BY s.nome, g.nome
            """)
            results = db.session.execute(relatorio_query).fetchall()
        else:
            relatorio_query = text("""
                SELECT 
                    s.id as silo_id,
                    s.nome as silo_nome,
                    s.capacidade_kg,
                    g.id as grao_id,
                    g.nome as grao_nome,
                    COALESCE(SUM(CASE WHEN ms.tipo_movimentacao = 'Entrada' THEN ms.quantidade_kg ELSE -ms.quantidade_kg END), 0) as estoque_grao
                FROM silo s
                LEFT JOIN movimentacao_silo ms ON s.id = ms.silo_id AND ms.user_id = :user_id
                LEFT JOIN grao g ON ms.grao_id = g.id
                WHERE s.user_id = :user_id
                GROUP BY s.id, s.nome, s.capacidade_kg, g.id, g.nome
                ORDER BY s.nome, g.nome
            """)
            results = db.session.execute(relatorio_query, {
                'user_id': current_user.id
            }).fetchall()

        # Processar resultados de forma eficiente
        silos_dict = {}
        for row in results:
            silo_id = row[0]
            if silo_id not in silos_dict:
                silos_dict[silo_id] = {
                    'silo': {
                        'id': row[0],
                        'nome': row[1],
                        'capacidade_kg': row[2]
                    },
                    'estoque_total': 0,
                    'graos': []
                }

            estoque_grao = float(row[5] or 0)
            if estoque_grao > 0 and row[3]:  # Se tem estoque e grão existe
                silos_dict[silo_id]['estoque_total'] += estoque_grao
                silos_dict[silo_id]['graos'].append({
                    'grao': {
                        'id': row[3],
                        'nome': row[4]
                    },
                    'estoque_kg':
                    estoque_grao,
                    'sacas':
                    round(estoque_grao / 60.0, 2)
                })

        # Calcular percentuais
        relatorio = []
        for silo_data in silos_dict.values():
            capacidade = silo_data['silo']['capacidade_kg']
            estoque_total = silo_data['estoque_total']
            percentual_ocupacao = (estoque_total / capacidade *
                                   100) if capacidade > 0 else 0
            capacidade_disponivel = capacidade - estoque_total

            relatorio.append({
                'silo':
                silo_data['silo'],
                'estoque_total':
                estoque_total,
                'percentual_ocupacao':
                round(percentual_ocupacao, 1),
                'capacidade_disponivel':
                capacidade_disponivel,
                'graos':
                silo_data['graos']
            })

        return render_template('relatorio_silos.html', relatorio=relatorio)

    except Exception as e:
        app.logger.error(f"Erro no relatório de silos: {str(e)}")
        flash('Erro ao gerar relatório. Tente novamente.', 'error')
        return render_template('relatorio_silos.html', relatorio=[])


@app.route('/api/silos/<int:silo_id>/capacidade-disponivel')
@login_required
def api_capacidade_disponivel(silo_id):
    """API para verificar capacidade disponível em tempo real"""
    try:
        silo = obter_por_id_usuario(Silo, silo_id)
        if not silo:
            return jsonify({'error': 'Silo não encontrado ou acesso negado'}), 404
        
        capacidade_disponivel = silo.get_capacidade_disponivel()
        return jsonify({
            'capacidade_disponivel': capacidade_disponivel,
            'capacidade_total': silo.capacidade_kg,
            'percentual_ocupacao': silo.get_percentual_ocupacao()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calcular-sacas/<float:peso_kg>')
@login_required
def api_calcular_sacas(peso_kg):
    """API para calcular número de sacas baseado no peso"""
    sacas = peso_kg / 60.0
    return jsonify({
        'peso_kg': peso_kg,
        'sacas': round(sacas, 2),
        'sacas_inteiras': int(sacas),
        'peso_restante_kg': peso_kg % 60
    })


# ROTAS DE TALHÕES
@app.route('/talhoes')
@login_required
def talhoes():
    """Página de gestão de talhões com mapa"""
    talhoes_list = filtrar_por_usuario(
        Talhao.query.order_by(Talhao.data_criacao.desc()),
        Talhao
    ).all()

    # Converter para JSON para o JavaScript
    talhoes_json = json.dumps([{
        'id': t.id,
        'nome': t.nome,
        'area_hectares': t.area_hectares,
        'area_alqueires': t.area_alqueires,
        'coordenadas': t.coordenadas,
        'cor': t.cor,
        'observacao': t.observacao
    } for t in talhoes_list])

    return render_template('talhoes.html',
                           talhoes=talhoes_list,
                           talhoes_json=talhoes_json)


@app.route('/talhoes/salvar', methods=['POST'])
@login_required
def salvar_talhao():
    """Salva ou atualiza um talhão"""
    try:
        # Aceitar tanto JSON quanto FormData
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        talhao_id = data.get('talhao_id')
        nome = data.get('nome', '').strip()
        coordenadas = data.get('coordenadas')
        area_hectares = float(data.get('area_hectares', 0))
        area_alqueires = float(data.get('area_alqueires', 0))
        cor = data.get('cor', '#FFD700')
        observacao = data.get('observacao', '').strip()

        if not nome:
            return jsonify({'success': False, 'error': 'Nome é obrigatório'}), 400

        if not coordenadas:
            return jsonify({'success': False, 'error': 'Coordenadas são obrigatórias'}), 400

        # Atualizar ou criar
        if talhao_id:
            talhao = obter_por_id_usuario(Talhao, int(talhao_id))
            if not talhao:
                return jsonify({'success': False, 'error': 'Talhão não encontrado'}), 404

            talhao.nome = nome
            talhao.coordenadas = coordenadas
            talhao.area_hectares = area_hectares
            talhao.area_alqueires = area_alqueires
            talhao.cor = cor
            talhao.observacao = observacao
            msg = 'Talhão atualizado com sucesso!'
        else:
            talhao = criar_com_usuario(Talhao,
                nome=nome,
                coordenadas=coordenadas,
                area_hectares=area_hectares,
                area_alqueires=area_alqueires,
                cor=cor,
                observacao=observacao
            )
            db.session.add(talhao)
            msg = 'Talhão cadastrado com sucesso!'

        db.session.commit()
        flash(msg, 'success')

        # Retornar dados do talhão para uso em AJAX
        return jsonify({
            'success': True,
            'talhao': {
                'id': talhao.id,
                'nome': talhao.nome,
                'area_hectares': talhao.area_hectares,
                'area_alqueires': talhao.area_alqueires
            }
        })

    except Exception as e:
        app.logger.error(f"Erro ao salvar talhão: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/talhoes/<int:id>/excluir', methods=['POST'])
@login_required
def excluir_talhao(id):
    """Exclui um talhão"""
    try:
        talhao = obter_por_id_usuario(Talhao, id)
        if not talhao:
            return jsonify({'success': False, 'error': 'Talhão não encontrado'}), 404

        db.session.delete(talhao)
        db.session.commit()

        flash(f'Talhão "{talhao.nome}" excluído com sucesso!', 'success')
        return jsonify({'success': True})

    except Exception as e:
        app.logger.error(f"Erro ao excluir talhão: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ROTAS DE CHUVA
@app.route('/chuva')
@login_required
def chuva():
    registros = filtrar_por_usuario(
        RegistroChuva.query, RegistroChuva).order_by(RegistroChuva.data.desc()).all()

    # Buscar talhões para seleção
    talhoes_list = filtrar_por_usuario(
        Talhao.query.order_by(Talhao.nome), Talhao
    ).all()

    return render_template('chuva.html', registros=registros, talhoes=talhoes_list)


@app.route('/chuva/adicionar', methods=['POST'])
@login_required
def adicionar_registro_chuva():
    try:
        data_str = request.form.get('data')
        milimetros = float(request.form.get('milimetros', 0))
        observacao = request.form.get('observacao', '')
        aplicar_todos = request.form.get('aplicar_todos') == 'true'
        talhoes_ids = request.form.getlist('talhoes_ids[]')  # Lista de IDs dos talhões selecionados

        if not data_str:
            flash('Data é obrigatória!', 'error')
            return redirect(url_for('chuva'))

        if milimetros < 0:
            flash('Quantidade de chuva não pode ser negativa!', 'error')
            return redirect(url_for('chuva'))

        # Validar seleção de talhões
        if not aplicar_todos and not talhoes_ids:
            flash('Selecione pelo menos um talhão ou marque "Aplicar a todos"!', 'error')
            return redirect(url_for('chuva'))

        data = datetime.strptime(data_str, '%Y-%m-%d').date()

        # Criar registro
        registro = criar_com_usuario(RegistroChuva,
                                     data=data,
                                     quantidade_mm=milimetros,
                                     observacao=observacao,
                                     aplicado_todos_talhoes=aplicar_todos)
        db.session.add(registro)
        db.session.flush()  # Para obter o ID do registro

        # Associar talhões se não for "todos"
        if not aplicar_todos and talhoes_ids:
            talhoes = filtrar_por_usuario(
                Talhao.query.filter(Talhao.id.in_(talhoes_ids)), Talhao
            ).all()
            registro.talhoes = talhoes

        db.session.commit()

        if aplicar_todos:
            flash('Registro de chuva adicionado para todos os talhões!', 'success')
        else:
            flash(f'Registro de chuva adicionado para {len(talhoes_ids)} talhão(ões)!', 'success')

    except ValueError as e:
        flash('Erro: Data ou quantidade inválida.', 'error')
    except Exception as e:
        flash('Erro ao adicionar registro de chuva. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar registro de chuva: {str(e)}")

    return redirect(url_for('chuva'))


@app.route('/chuva/editar/<int:id>', methods=['POST'])
@login_required
def editar_registro_chuva(id):
    try:
        registro = obter_por_id_usuario(RegistroChuva, id)
        if not registro:
            flash('Registro não encontrado ou acesso negado!', 'error')
            return redirect(url_for('chuva'))

        data_str = request.form.get('data')
        milimetros = float(request.form.get('milimetros', 0))
        observacao = request.form.get('observacao', '')

        if not data_str:
            flash('Data é obrigatória!', 'error')
            return redirect(url_for('chuva'))

        if milimetros < 0:
            flash('Quantidade de chuva não pode ser negativa!', 'error')
            return redirect(url_for('chuva'))

        data = datetime.strptime(data_str, '%Y-%m-%d').date()

        # Verificar se já existe outro registro para esta data
        registro_existente = filtrar_por_usuario(
            RegistroChuva.query.filter_by(data=data).filter(RegistroChuva.id != id),
            RegistroChuva).first()
        if registro_existente:
            flash('Já existe um registro para esta data!', 'error')
            return redirect(url_for('chuva'))

        registro.data = data
        registro.quantidade_mm = milimetros
        registro.observacao = observacao

        db.session.commit()
        flash('Registro de chuva atualizado com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Data ou quantidade inválida.', 'error')
    except Exception as e:
        flash('Erro ao atualizar registro de chuva. Tente novamente.', 'error')
        app.logger.error(f"Erro ao editar registro de chuva: {str(e)}")

    return redirect(url_for('chuva'))


@app.route('/chuva/excluir/<int:id>', methods=['POST'])
@login_required
def excluir_registro_chuva(id):
    try:
        registro = obter_por_id_usuario(RegistroChuva, id)
        if not registro:
            flash('Registro não encontrado ou acesso negado!', 'error')
            return redirect(url_for('chuva'))
        
        db.session.delete(registro)
        db.session.commit()
        flash('Registro de chuva excluído com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao excluir registro de chuva. Tente novamente.', 'error')
        app.logger.error(f"Erro ao excluir registro de chuva: {str(e)}")

    return redirect(url_for('chuva'))


# ROTAS DE API


# ROTAS DE RELATÓRIOS
@app.route('/relatorios')
@login_required
def relatorios():
    """Página unificada de relatórios com visão geral simples"""
    try:
        # Dados consolidados básicos com isolamento de usuário
        total_silos = contar_registros_usuario(Silo)
        total_funcionarios = contar_registros_usuario(Funcionario)
        total_insumos = contar_registros_usuario(Insumo)
        total_maquinas = contar_registros_usuario(Maquinario)

        # Dados dos silos com isolamento de usuário
        silos = filtrar_por_usuario(Silo.query, Silo).all()
        capacidade_total = sum(silo.capacidade_kg for silo in silos)
        estoque_total = sum(silo.get_estoque_total() for silo in silos)
        ocupacao_percentual = (estoque_total / capacidade_total *
                               100) if capacidade_total > 0 else 0

        # Movimentações recentes com isolamento de usuário (últimas 10)
        movimentacoes_recentes = filtrar_por_usuario(
            MovimentacaoSilo.query, MovimentacaoSilo).order_by(
            MovimentacaoSilo.data_movimentacao.desc()).limit(10).all()

        # Insumos com baixo estoque com isolamento de usuário
        insumos_baixo_estoque = filtrar_por_usuario(
            Insumo.query.filter(Insumo.quantidade < 50), Insumo).limit(5).all()

        dados = {
            'resumo': {
                'total_silos': total_silos,
                'total_funcionarios': total_funcionarios,
                'total_insumos': total_insumos,
                'total_maquinas': total_maquinas,
                'capacidade_total': capacidade_total,
                'estoque_total': estoque_total,
                'ocupacao_percentual': ocupacao_percentual
            },
            'silos': silos,
            'movimentacoes_recentes': movimentacoes_recentes,
            'insumos_baixo_estoque': insumos_baixo_estoque
        }

        return render_template('relatorios.html', dados=dados)

    except Exception as e:
        app.logger.error(f"Erro ao gerar relatórios: {str(e)}")
        flash('Erro ao carregar relatórios. Tente novamente.', 'error')
        return redirect(url_for('dashboard'))


# ROTAS FINANCEIRAS FUTURAS (PLACEHOLDERS)
@app.route('/contas-receber')
@login_required
def contas_receber():
    """Página principal de contas a receber"""
    contas = filtrar_por_usuario(
        ContasReceber.query.order_by(ContasReceber.data_vencimento.desc()),
        ContasReceber
    ).all()

    # Estatísticas
    total_pendente = sum(c.valor for c in contas if c.status == 'pendente')
    total_vencido = sum(c.valor for c in contas if c.esta_vencida())
    total_recebido = sum(c.valor for c in contas if c.status == 'recebido')

    return render_template('contas_receber.html',
                         contas=contas,
                         total_pendente=total_pendente,
                         total_vencido=total_vencido,
                         total_recebido=total_recebido,
                         datetime=datetime)


@app.route('/contas-receber/adicionar', methods=['POST'])
@login_required
def adicionar_conta_receber():
    try:
        descricao = request.form.get('descricao', '').strip()
        cliente = request.form.get('cliente', '').strip()
        categoria = request.form.get('categoria', '').strip()
        valor = float(request.form.get('valor', 0))
        data_vencimento_str = request.form.get('data_vencimento')
        numero_documento = request.form.get('numero_documento', '').strip()
        observacoes = request.form.get('observacoes', '').strip()

        if not descricao or not cliente or not categoria:
            flash('Descrição, cliente e categoria são obrigatórios!', 'error')
            return redirect(url_for('contas_receber'))

        if valor <= 0:
            flash('Valor deve ser maior que zero!', 'error')
            return redirect(url_for('contas_receber'))

        if not data_vencimento_str:
            flash('Data de vencimento é obrigatória!', 'error')
            return redirect(url_for('contas_receber'))

        data_vencimento = datetime.strptime(data_vencimento_str, '%Y-%m-%d').date()

        conta = criar_com_usuario(
            ContasReceber,
            descricao=descricao,
            cliente=cliente,
            categoria=categoria,
            valor=valor,
            data_vencimento=data_vencimento,
            numero_documento=numero_documento if numero_documento else None,
            observacoes=observacoes if observacoes else None
        )

        db.session.add(conta)
        db.session.commit()

        flash('Conta a receber adicionada com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Verifique os valores fornecidos.', 'error')
        app.logger.error(f"Erro ao adicionar conta a receber: {str(e)}")
    except Exception as e:
        flash('Erro ao adicionar conta. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar conta a receber: {str(e)}")

    return redirect(url_for('contas_receber'))


@app.route('/contas-receber/editar/<int:id>', methods=['POST'])
@login_required
def editar_conta_receber(id):
    try:
        conta = get_or_404_user_scoped(ContasReceber, id)

        descricao = request.form.get('descricao', '').strip()
        cliente = request.form.get('cliente', '').strip()
        categoria = request.form.get('categoria', '').strip()
        valor = float(request.form.get('valor', 0))
        data_vencimento_str = request.form.get('data_vencimento')
        numero_documento = request.form.get('numero_documento', '').strip()
        observacoes = request.form.get('observacoes', '').strip()

        if not descricao or not cliente or not categoria:
            flash('Descrição, cliente e categoria são obrigatórios!', 'error')
            return redirect(url_for('contas_receber'))

        if valor <= 0:
            flash('Valor deve ser maior que zero!', 'error')
            return redirect(url_for('contas_receber'))

        conta.descricao = descricao
        conta.cliente = cliente
        conta.categoria = categoria
        conta.valor = valor
        conta.numero_documento = numero_documento if numero_documento else None
        conta.observacoes = observacoes if observacoes else None

        if data_vencimento_str:
            conta.data_vencimento = datetime.strptime(data_vencimento_str, '%Y-%m-%d').date()

        db.session.commit()
        flash('Conta atualizada com sucesso!', 'success')
    except ValueError as e:
        flash('Erro: Verifique os valores fornecidos.', 'error')
        app.logger.error(f"Erro ao editar conta: {str(e)}")
    except Exception as e:
        flash('Erro ao atualizar conta. Tente novamente.', 'error')
        app.logger.error(f"Erro ao editar conta: {str(e)}")

    return redirect(url_for('contas_receber'))


@app.route('/contas-receber/excluir/<int:id>', methods=['POST'])
@login_required
def excluir_conta_receber(id):
    try:
        conta = obter_por_id_usuario(ContasReceber, id)
        if not conta:
            flash('Conta não encontrada ou acesso negado!', 'error')
            return redirect(url_for('contas_receber'))

        db.session.delete(conta)
        db.session.commit()
        flash('Conta excluída com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao excluir conta. Tente novamente.', 'error')
        app.logger.error(f"Erro ao excluir conta: {str(e)}")

    return redirect(url_for('contas_receber'))


@app.route('/contas-receber/receber/<int:id>', methods=['POST'])
@login_required
def receber_conta(id):
    try:
        conta = get_or_404_user_scoped(ContasReceber, id)

        if conta.status == 'recebido':
            flash('Esta conta já foi recebida!', 'warning')
            return redirect(url_for('contas_receber'))

        forma_recebimento = request.form.get('forma_recebimento', '').strip()
        data_recebimento_str = request.form.get('data_recebimento')

        if not forma_recebimento:
            flash('Forma de recebimento é obrigatória!', 'error')
            return redirect(url_for('contas_receber'))

        conta.status = 'recebido'
        conta.forma_recebimento = forma_recebimento

        if data_recebimento_str:
            conta.data_recebimento = datetime.strptime(data_recebimento_str, '%Y-%m-%d').date()
        else:
            conta.data_recebimento = datetime.now().date()

        db.session.commit()
        flash('Recebimento registrado com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao registrar recebimento. Tente novamente.', 'error')
        app.logger.error(f"Erro ao receber conta: {str(e)}")

    return redirect(url_for('contas_receber'))


@app.route('/contas-receber/cancelar/<int:id>', methods=['POST'])
@login_required
def cancelar_conta_receber(id):
    try:
        conta = get_or_404_user_scoped(ContasReceber, id)

        if conta.status == 'recebido':
            flash('Não é possível cancelar uma conta já recebida!', 'error')
            return redirect(url_for('contas_receber'))

        conta.status = 'cancelado'
        db.session.commit()
        flash('Conta cancelada com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao cancelar conta. Tente novamente.', 'error')
        app.logger.error(f"Erro ao cancelar conta: {str(e)}")

    return redirect(url_for('contas_receber'))


@app.route('/fluxo-caixa')
@login_required
def fluxo_caixa():
    """Página de Fluxo de Caixa com consolidação de receitas e despesas"""
    # Obter período (padrão: últimos 6 meses)
    periodo = request.args.get('periodo', '6')
    try:
        meses = int(periodo)
    except:
        meses = 6

    data_inicio = datetime.now() - timedelta(days=meses*30)

    # Buscar contas a pagar
    contas_pagar = filtrar_por_usuario(
        ContasPagar.query.filter(ContasPagar.data_vencimento >= data_inicio),
        ContasPagar
    ).all()

    # Buscar contas a receber
    contas_receber = filtrar_por_usuario(
        ContasReceber.query.filter(ContasReceber.data_vencimento >= data_inicio),
        ContasReceber
    ).all()

    # Calcular totais do mês atual
    hoje = datetime.now().date()
    primeiro_dia_mes = hoje.replace(day=1)
    if hoje.month == 12:
        primeiro_dia_proximo_mes = hoje.replace(year=hoje.year + 1, month=1, day=1)
    else:
        primeiro_dia_proximo_mes = hoje.replace(month=hoje.month + 1, day=1)

    # Entradas do mês (contas recebidas)
    entradas_mes = sum(
        c.valor for c in contas_receber
        if c.status == 'recebido'
        and c.data_recebimento
        and primeiro_dia_mes <= c.data_recebimento < primeiro_dia_proximo_mes
    )

    # Saídas do mês (contas pagas)
    saidas_mes = sum(
        c.valor for c in contas_pagar
        if c.status == 'pago'
        and c.data_pagamento
        and primeiro_dia_mes <= c.data_pagamento < primeiro_dia_proximo_mes
    )

    # Saldo do mês
    saldo_mes = entradas_mes - saidas_mes

    # Contas pendentes
    total_pendente_receber = sum(c.valor for c in contas_receber if c.status == 'pendente')
    total_pendente_pagar = sum(c.valor for c in contas_pagar if c.status == 'pendente')
    saldo_projetado = total_pendente_receber - total_pendente_pagar

    # Preparar dados para gráfico (últimos 6 meses)
    meses_grafico = []
    entradas_grafico = []
    saidas_grafico = []

    for i in range(meses, -1, -1):
        data_ref = datetime.now() - timedelta(days=i*30)
        mes_ano = data_ref.strftime('%m/%Y')

        # Primeiro e último dia do mês
        primeiro_dia = data_ref.replace(day=1).date()
        if data_ref.month == 12:
            ultimo_dia = data_ref.replace(year=data_ref.year + 1, month=1, day=1).date()
        else:
            ultimo_dia = data_ref.replace(month=data_ref.month + 1, day=1).date()

        # Calcular entradas do mês
        entradas = sum(
            c.valor for c in contas_receber
            if c.status == 'recebido'
            and c.data_recebimento
            and primeiro_dia <= c.data_recebimento < ultimo_dia
        )

        # Calcular saídas do mês
        saidas = sum(
            c.valor for c in contas_pagar
            if c.status == 'pago'
            and c.data_pagamento
            and primeiro_dia <= c.data_pagamento < ultimo_dia
        )

        meses_grafico.append(mes_ano)
        entradas_grafico.append(entradas)
        saidas_grafico.append(saidas)

    # Movimentações recentes (últimas 10)
    movimentacoes_recentes = []

    for conta in contas_pagar:
        if conta.status == 'pago' and conta.data_pagamento:
            movimentacoes_recentes.append({
                'data': conta.data_pagamento,
                'descricao': conta.descricao,
                'tipo': 'saida',
                'valor': conta.valor,
                'categoria': conta.categoria
            })

    for conta in contas_receber:
        if conta.status == 'recebido' and conta.data_recebimento:
            movimentacoes_recentes.append({
                'data': conta.data_recebimento,
                'descricao': conta.descricao,
                'tipo': 'entrada',
                'valor': conta.valor,
                'categoria': conta.categoria
            })

    # Ordenar por data (mais recentes primeiro)
    movimentacoes_recentes.sort(key=lambda x: x['data'], reverse=True)
    movimentacoes_recentes = movimentacoes_recentes[:10]

    return render_template('fluxo_caixa.html',
                         entradas_mes=entradas_mes,
                         saidas_mes=saidas_mes,
                         saldo_mes=saldo_mes,
                         total_pendente_receber=total_pendente_receber,
                         total_pendente_pagar=total_pendente_pagar,
                         saldo_projetado=saldo_projetado,
                         meses_grafico=json.dumps(meses_grafico),
                         entradas_grafico=json.dumps(entradas_grafico),
                         saidas_grafico=json.dumps(saidas_grafico),
                         movimentacoes_recentes=movimentacoes_recentes,
                         periodo_selecionado=meses)


@app.route('/contas-bancarias')
@login_required
def contas_bancarias():
    """Página de Contas Bancárias - Em desenvolvimento"""
    return render_template('contas_bancarias.html')


# ROTAS DE ADMINISTRAÇÃO
@app.route('/administracao')
@admin_required
def administracao():
    """Página principal de administração"""
    usuarios = Usuario.query.all()
    return render_template('administracao.html', usuarios=usuarios)


@app.route('/administracao/usuario/adicionar', methods=['POST'])
@admin_required
def adicionar_usuario():
    """Adicionar novo usuário manualmente pelo administrador"""
    try:
        nome_completo = request.form.get('nome_completo', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        plano = request.form.get('plano', 'basic')
        is_admin = request.form.get('is_admin') == 'on'
        
        # Validações
        errors = []
        
        if not nome_completo or len(nome_completo) < 3:
            errors.append('Nome completo deve ter pelo menos 3 caracteres.')
            
        if not username or len(username) < 3:
            errors.append('Nome de usuário deve ter pelo menos 3 caracteres.')
            
        if not email or '@' not in email:
            errors.append('E-mail inválido.')
            
        if not password or len(password) < 6:
            errors.append('Senha deve ter pelo menos 6 caracteres.')

        # Verificar se usuário já existe
        user_exists = Usuario.query.filter_by(username=username).first()
        if user_exists:
            errors.append('Nome de usuário já está em uso.')
            
        # Verificar se email já existe
        email_exists = Usuario.query.filter_by(email=email).first()
        if email_exists:
            errors.append('E-mail já está cadastrado.')

        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('gerencia_usuarios'))

        # Criar novo usuário com acesso imediato
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

        # Calcular data de expiração (30 dias a partir de hoje)
        data_expiracao = (datetime.utcnow() + timedelta(days=30)).date()

        new_user = Usuario(
            username=username,
            password_hash=password_hash,
            nome_completo=nome_completo,
            email=email,
            status_aprovacao='aprovado',  # Aprovado automaticamente
            data_cadastro=datetime.utcnow(),
            data_aprovacao=datetime.utcnow(),
            aprovado_por_id=current_user.id,
            primeiro_acesso=False,  # Não forçar troca de senha
            email_verificado=True,  # Email verificado automaticamente
            plano=plano,
            is_admin=is_admin,
            status_assinatura='ativa',  # Garantir que está ativo
            data_inicio_teste=datetime.utcnow(),  # Iniciar período de teste
            data_expiracao=data_expiracao,  # Expiração em 30 dias
            user_role='cliente'  # Definir role padrão
        )

        db.session.add(new_user)

        try:
            db.session.commit()
            app.logger.info(f"✅ Usuário criado manualmente pelo admin {current_user.username}: {username} ({email}) - Todos os campos definidos")
            flash(f'Usuário {username} criado com sucesso! Acesso liberado imediatamente.', 'success')
        except Exception as commit_error:
            db.session.rollback()
            app.logger.error(f"❌ ERRO ao fazer commit do usuário {username}: {str(commit_error)}")
            flash(f'Erro ao criar usuário: {str(commit_error)}', 'error')
            raise
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erro ao criar usuário manualmente: {str(e)}")
        flash('Erro interno. Tente novamente mais tarde.', 'error')
    
    return redirect(url_for('gerencia_usuarios'))


@app.route('/administracao/usuario/editar/<int:user_id>', methods=['POST'])
@admin_required
def editar_usuario(user_id):
    """Editar senha de usuário"""
    try:
        # NOTA: Administradores podem editar qualquer usuário
        usuario = Usuario.query.get_or_404(user_id)
        nova_senha = request.form.get('nova_senha')

        if not nova_senha:
            flash('Nova senha é obrigatória!', 'error')
            return redirect(url_for('administracao'))

        usuario.password_hash = bcrypt.generate_password_hash(
            nova_senha).decode('utf-8')
        db.session.commit()

        flash('Senha atualizada com sucesso!', 'success')
    except Exception as e:
        flash('Erro ao atualizar senha. Tente novamente.', 'error')
        app.logger.error(f"Erro ao editar usuário: {str(e)}")

    return redirect(url_for('gerencia_usuarios'))


# NOVAS ROTAS DE ADMINISTRAÇÃO CATEGORIZADA
@app.route('/administracao/usuarios')
@admin_required
def gerencia_usuarios():
    """Gerência de usuários com aprovações pendentes"""
    usuarios = Usuario.query.all()
    usuarios_pendentes = Usuario.query.filter_by(status_aprovacao='pendente').all()
    
    
    return render_template('gerencia_usuarios.html', 
                         usuarios=usuarios, 
                         usuarios_pendentes=usuarios_pendentes)


@app.route('/administracao/usuario/aprovar/<int:user_id>', methods=['POST'])
@admin_required
def aprovar_usuario(user_id):
    """Aprovar usuário pendente"""
    try:
        usuario = Usuario.query.get_or_404(user_id)
        
        if usuario.status_aprovacao != 'pendente':
            flash('Usuário já foi processado.', 'warning')
            return redirect(url_for('gerencia_usuarios'))
        
        usuario.aprovar(current_user.id)
        db.session.commit()
        
        app.logger.info(f"Usuário {usuario.username} aprovado por {current_user.username}")
        flash(f'Usuário {usuario.nome_completo} ({usuario.username}) aprovado com sucesso!', 'success')
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erro ao aprovar usuário: {str(e)}")
        flash('Erro ao aprovar usuário. Tente novamente.', 'error')
    
    return redirect(url_for('gerencia_usuarios'))


@app.route('/administracao/usuario/rejeitar/<int:user_id>', methods=['POST'])
@admin_required
def rejeitar_usuario(user_id):
    """Rejeitar usuário pendente"""
    try:
        usuario = Usuario.query.get_or_404(user_id)
        
        if usuario.status_aprovacao != 'pendente':
            flash('Usuário já foi processado.', 'warning')
            return redirect(url_for('gerencia_usuarios'))
        
        usuario.rejeitar(current_user.id)
        db.session.commit()
        
        app.logger.info(f"Usuário {usuario.username} rejeitado por {current_user.username}")
        flash(f'Usuário {usuario.nome_completo} ({usuario.username}) rejeitado.', 'info')
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erro ao rejeitar usuário: {str(e)}")
        flash('Erro ao rejeitar usuário. Tente novamente.', 'error')
    
    return redirect(url_for('gerencia_usuarios'))


@app.route('/administracao/usuario/excluir/<int:user_id>', methods=['POST'])
@admin_required
def excluir_usuario(user_id):
    """Excluir usuário do sistema"""
    try:
        usuario = Usuario.query.get_or_404(user_id)
        
        # Não permitir que admin exclua outro admin (exceto a si mesmo)
        if usuario.is_admin and usuario.id != current_user.id:
            flash('Não é possível excluir outro administrador.', 'danger')
            return redirect(url_for('gerencia_usuarios'))
        
        # Não permitir autoexclusão de administradores
        if usuario.id == current_user.id:
            flash('Não é possível excluir sua própria conta por segurança.', 'danger')
            return redirect(url_for('gerencia_usuarios'))
        
        # Verificar se não é o último admin do sistema
        if usuario.is_admin:
            total_admins = Usuario.query.filter_by(is_admin=True).count()
            if total_admins <= 1:
                flash('Não é possível excluir o último administrador do sistema.', 'danger')
                return redirect(url_for('gerencia_usuarios'))
        
        username = usuario.username
        nome = usuario.nome_completo or usuario.username
        
        # TODO: Aqui poderia implementar exclusão em cascata dos dados relacionados
        # Por agora, vamos só excluir o usuário
        
        db.session.delete(usuario)
        db.session.commit()
        
        app.logger.info(f"Usuário excluído pelo admin {current_user.username}: {username}")
        flash(f'Usuário {nome} ({username}) foi excluído com sucesso.', 'success')
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erro ao excluir usuário: {str(e)}")
        flash('Erro ao excluir usuário. Tente novamente.', 'error')
    
    return redirect(url_for('gerencia_usuarios'))


@app.route('/administracao/usuario/alterar-permissoes/<int:user_id>', methods=['POST'])
@super_admin_required
def alterar_permissoes_usuario(user_id):
    """Alterar permissões de usuário (apenas Super Admin)"""
    try:
        usuario = Usuario.query.get_or_404(user_id)
        
        # Verificar se não está tentando alterar as próprias permissões
        if usuario.id == current_user.id:
            flash('Não é possível alterar suas próprias permissões por segurança.', 'danger')
            return redirect(url_for('gerencia_usuarios'))
        
        # Obter valores do formulário
        novo_user_role = request.form.get('user_role')
        novo_is_admin = request.form.get('is_admin') == 'true'
        
        # Validar user_role
        roles_validos = ['super_admin', 'cliente', 'funcionario']
        if novo_user_role not in roles_validos:
            flash('Nível de permissão inválido.', 'error')
            return redirect(url_for('gerencia_usuarios'))
        
        # Verificar se não está removendo o último super_admin do sistema
        if usuario.user_role == 'super_admin' and novo_user_role != 'super_admin':
            total_super_admins = Usuario.query.filter_by(user_role='super_admin').count()
            if total_super_admins <= 1:
                flash('Não é possível rebaixar o último Super Administrador do sistema por segurança.', 'danger')
                return redirect(url_for('gerencia_usuarios'))
        
        # Salvar valores antigos para log
        role_antigo = usuario.user_role
        admin_antigo = usuario.is_admin
        
        # Atualizar permissões
        usuario.user_role = novo_user_role
        usuario.is_admin = novo_is_admin
        
        db.session.commit()
        
        # Log da alteração
        permissao_nova = f"{novo_user_role}" + (" + Admin" if novo_is_admin else "")
        permissao_antiga = f"{role_antigo}" + (" + Admin" if admin_antigo else "")
        
        app.logger.info(f"Permissões alteradas pelo super admin {current_user.username}: "
                       f"Usuário {usuario.username} de '{permissao_antiga}' para '{permissao_nova}'")
        
        flash(f'Permissões do usuário {usuario.nome_completo} ({usuario.username}) '
              f'alteradas para: {permissao_nova}', 'success')
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erro ao alterar permissões do usuário: {str(e)}")
        flash('Erro ao alterar permissões. Tente novamente.', 'error')
    
    return redirect(url_for('gerencia_usuarios'))


@app.route('/administracao/usuario/resetar-senha/<int:user_id>', methods=['POST'])
@admin_required
def resetar_senha_admin(user_id):
    """Resetar senha de usuário pelo administrador usando token seguro"""
    try:
        usuario = Usuario.query.get_or_404(user_id)
        
        # Não permitir que admin resete senha de outro admin
        if usuario.is_admin and usuario.id != current_user.id:
            flash('Não é possível resetar senha de outro administrador.', 'danger')
            return redirect(url_for('gerencia_usuarios'))
        
        # Gerar novo token de reset e invalidar senha atual
        import secrets
        senha_temporaria = secrets.token_urlsafe(32)
        usuario.password_hash = bcrypt.generate_password_hash(senha_temporaria).decode('utf-8')
        token_reset = usuario.gerar_token_verificacao()
        usuario.primeiro_acesso = True  # Forçar redefinição de senha
        usuario.email_verificado = False  # Requerer nova verificação
        
        db.session.commit()
        
        # Enviar email com link de reset
        email_enviado = enviar_email_reset_admin(usuario.email, usuario.nome_completo, token_reset)
        
        app.logger.info(f"Senha resetada pelo admin {current_user.username} para usuário {usuario.username}")
        
        if email_enviado:
            flash(f'Link de reset de senha enviado para {usuario.email}. O usuário deve verificar o e-mail.', 'success')
        else:
            flash(f'Reset iniciado, mas erro no envio do e-mail. Informe o usuário: {usuario.email}', 'warning')
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erro ao resetar senha: {str(e)}")
        flash('Erro ao resetar senha. Tente novamente.', 'error')
    
    return redirect(url_for('gerencia_usuarios'))


@app.route('/administracao/import-dados')
@admin_required
def import_dados():
    """Página de import de dados de ERP"""
    return render_template('import_dados.html')


# Configuração de segurança para uploads
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'txt', 'csv'}
UPLOAD_FOLDER = 'uploads'

def allowed_file(filename):
    """Verifica se o arquivo tem extensão permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_filename_custom(filename):
    """Cria nome de arquivo seguro"""
    import re
    import os
    # Remove caracteres perigosos e mantem apenas letras, números, pontos e hífens
    filename = re.sub(r'[^\w\s.-]', '', filename)
    # Substitui espaços por underscore
    filename = re.sub(r'\s+', '_', filename)
    # Limita o tamanho do nome
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    return name + ext

@app.route('/administracao/upload-dados', methods=['POST'])
@admin_required
def upload_dados():
    """Processar upload de dados de ERP com segurança aprimorada"""
    try:
        if 'arquivo' not in request.files:
            flash('Nenhum arquivo selecionado!', 'error')
            return redirect(url_for('import_dados'))
        
        arquivo = request.files['arquivo']
        if arquivo.filename == '':
            flash('Nenhum arquivo selecionado!', 'error')
            return redirect(url_for('import_dados'))
        
        # Verificar se arquivo tem extensão permitida
        if not allowed_file(arquivo.filename):
            flash('Formato de arquivo não suportado! Use apenas: .xlsx, .xls, .txt ou .csv', 'error')
            return redirect(url_for('import_dados'))
        
        # Verificar tamanho do arquivo
        arquivo.seek(0, 2)  # Mover para o final
        file_size = arquivo.tell()
        arquivo.seek(0)  # Voltar ao início
        
        if file_size > MAX_FILE_SIZE:
            flash(f'Arquivo muito grande! Tamanho máximo: {MAX_FILE_SIZE // (1024*1024)}MB', 'error')
            return redirect(url_for('import_dados'))
        
        # Verificar se é realmente um arquivo (não vazio)
        if file_size == 0:
            flash('O arquivo está vazio!', 'error')
            return redirect(url_for('import_dados'))
        
        # Criar pasta de upload se não existir
        import os
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER, mode=0o755)
        
        # Criar nome de arquivo seguro com timestamp
        original_filename = secure_filename_custom(arquivo.filename)
        extensao = original_filename.rsplit('.', 1)[1].lower()
        filename = f"import_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extensao}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Salvar arquivo
        arquivo.save(filepath)
        
        # Log da operação para auditoria
        app.logger.info(f"Upload realizado por {current_user.username} (ID: {current_user.id}): {original_filename} -> {filepath}")
        
        flash(f'Arquivo "{original_filename}" carregado com sucesso! Processamento em desenvolvimento.', 'success')
        
    except Exception as e:
        flash('Erro ao processar arquivo. Tente novamente.', 'error')
        app.logger.error(f"Erro no upload de dados: {str(e)}")
    
    return redirect(url_for('import_dados'))


@app.route('/administracao/modelo-template')
@admin_required
def download_template():
    """Download do modelo padrão para import de dados"""
    try:
        import io
        from flask import send_file
        
        # Criar arquivo de template simples
        template_content = """# MODELO DE IMPORT DE DADOS - FAZENDA REBELATO
# Formato: TXT (separado por vírgulas)
# 
# INSUMOS GERAIS:
# nome,tipo,quantidade,unidade,preco_unitario,fornecedor,observacoes
# Exemplo: Ração Bovina,Alimentação,1000,kg,2.50,Fornecedor ABC,Ração de alta qualidade
#
# FUNCIONÁRIOS:
# nome,cargo,salario,data_admissao,telefone,email
# Exemplo: João Silva,Operador,2500.00,2024-01-15,11999887766,joao@email.com
#
# SILOS:
# nome,capacidade,localizacao,tipo_grao,observacoes
# Exemplo: Silo 1,5000,Galpão A,Milho,Silo principal de armazenamento
#
# EQUIPAMENTOS:
# nome,tipo,modelo,ano,valor,status,observacoes
# Exemplo: Trator John Deere,Trator,5075E,2023,150000.00,Ativo,Trator para plantio
#
# Formato XLSX também suportado com as mesmas colunas em planilhas separadas
"""
        
        # Criar arquivo em memória
        output = io.BytesIO()
        output.write(template_content.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name='modelo_import_fazenda_rebelato.txt',
            mimetype='text/plain'
        )
        
    except Exception as e:
        flash('Erro ao gerar modelo. Tente novamente.', 'error')
        app.logger.error(f"Erro ao gerar template: {str(e)}")
        return redirect(url_for('import_dados'))


# ===== ROTAS PARA GERENCIAMENTO DE CLIENTES (ADMIN) =====

@app.route('/admin/clientes')
@admin_required
def gerenciar_clientes():
    """Página principal de gerenciamento de clientes"""
    try:
        # Administradores veem todos os clientes (sem filtro de usuário)
        clientes = Cliente.query.order_by(Cliente.data_cadastro.desc()).all()
        return render_template('gerenciar_clientes.html', clientes=clientes)
    except Exception as e:
        flash('Erro ao carregar clientes.', 'error')
        app.logger.error(f"Erro ao carregar clientes: {str(e)}")
        return redirect(url_for('dashboard'))

@app.route('/admin/clientes/adicionar', methods=['POST'])
@admin_required
def adicionar_cliente():
    """Adicionar novo cliente"""
    try:
        nome = request.form.get('nome')
        categoria = request.form.get('categoria')
        
        if not nome or not categoria:
            flash('Nome e categoria são obrigatórios!', 'error')
            return redirect(url_for('gerenciar_clientes'))
        
        # Verificar se cliente já existe para este usuário
        user_id_destino = request.form.get('user_id', current_user.id)
        if Cliente.query.filter_by(nome=nome, user_id=user_id_destino).first():
            flash('Cliente com este nome já existe!', 'error')
            return redirect(url_for('gerenciar_clientes'))
        
        # Criar novo cliente
        novo_cliente = criar_com_usuario(
            Cliente,
            nome=nome,
            categoria=categoria,
            empresa=request.form.get('empresa', ''),
            documento=request.form.get('documento', ''),
            telefone=request.form.get('telefone', ''),
            email=request.form.get('email', ''),
            endereco=request.form.get('endereco', ''),
            status=request.form.get('status', 'Ativo'),
            valor_total=float(request.form.get('valor_inicial', 0)),
            observacoes=request.form.get('observacoes', ''),
            target_user_id=user_id_destino
        )
        
        db.session.add(novo_cliente)
        db.session.commit()
        
        flash('Cliente adicionado com sucesso!', 'success')
        
    except ValueError as e:
        flash('Erro nos dados fornecidos.', 'error')
        app.logger.error(f"Erro de validação ao adicionar cliente: {str(e)}")
    except Exception as e:
        db.session.rollback()
        flash('Erro ao adicionar cliente. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar cliente: {str(e)}")
    
    return redirect(url_for('gerenciar_clientes'))

@app.route('/admin/clientes/<int:cliente_id>/detalhes')
@admin_required
def detalhes_cliente(cliente_id):
    """Retorna detalhes completos do cliente via AJAX"""
    try:
        # Administrador pode ver qualquer cliente
        cliente = Cliente.query.get_or_404(cliente_id)
        
        # Buscar transações do cliente
        transacoes = TransacaoCliente.query.filter_by(cliente_id=cliente_id)\
                                         .order_by(TransacaoCliente.data_transacao.desc())\
                                         .limit(10).all()
        
        # Calcular estatísticas
        total_recebimentos = db.session.query(func.sum(TransacaoCliente.valor))\
                                     .filter_by(cliente_id=cliente_id, tipo='Recebimento').scalar() or 0
        total_pagamentos = db.session.query(func.sum(TransacaoCliente.valor))\
                                   .filter_by(cliente_id=cliente_id, tipo='Pagamento').scalar() or 0
        
        html_content = f"""
        <div class="row">
            <div class="col-md-6">
                <h6>Informações do Cliente</h6>
                <table class="table table-sm">
                    <tr><th>Nome:</th><td>{cliente.nome}</td></tr>
                    <tr><th>Empresa:</th><td>{cliente.empresa or '-'}</td></tr>
                    <tr><th>Categoria:</th><td>{cliente.categoria}</td></tr>
                    <tr><th>Documento:</th><td>{cliente.documento or '-'}</td></tr>
                    <tr><th>Telefone:</th><td>{cliente.telefone or '-'}</td></tr>
                    <tr><th>Email:</th><td>{cliente.email or '-'}</td></tr>
                    <tr><th>Status:</th><td><span class="badge bg-{'success' if cliente.status == 'Ativo' else 'danger'}">{cliente.status}</span></td></tr>
                </table>
            </div>
            <div class="col-md-6">
                <h6>Resumo Financeiro</h6>
                <table class="table table-sm">
                    <tr><th>Total Recebimentos:</th><td class="text-success">R$ {total_recebimentos:,.2f}</td></tr>
                    <tr><th>Total Pagamentos:</th><td class="text-danger">R$ {total_pagamentos:,.2f}</td></tr>
                    <tr><th>Saldo Atual:</th><td class="text-{'success' if cliente.valor_total >= 0 else 'danger'}">R$ {cliente.valor_total:,.2f}</td></tr>
                    <tr><th>Última Transação:</th><td>{cliente.ultima_transacao.strftime('%d/%m/%Y') if cliente.ultima_transacao else '-'}</td></tr>
                </table>
            </div>
        </div>
        
        <h6 class="mt-3">Últimas Transações</h6>
        <div class="table-responsive">
            <table class="table table-sm">
                <thead>
                    <tr>
                        <th>Data</th>
                        <th>Tipo</th>
                        <th>Valor</th>
                        <th>Descrição</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for transacao in transacoes:
            html_content += f"""
                    <tr>
                        <td>{transacao.data_transacao.strftime('%d/%m/%Y %H:%M')}</td>
                        <td><span class="badge bg-{'success' if transacao.tipo == 'Recebimento' else 'primary'}">{transacao.tipo}</span></td>
                        <td class="text-{'success' if transacao.tipo == 'Recebimento' else 'danger'}">R$ {transacao.valor:,.2f}</td>
                        <td>{transacao.descricao}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </div>
        """
        
        return html_content
        
    except Exception as e:
        app.logger.error(f"Erro ao carregar detalhes do cliente: {str(e)}")
        return "<p class='text-danger'>Erro ao carregar detalhes do cliente.</p>"

@app.route('/admin/clientes/<int:cliente_id>/transacao', methods=['POST'])
@admin_required
def adicionar_transacao_cliente(cliente_id):
    """Adicionar nova transação para cliente"""
    try:
        # Administrador pode adicionar transação para qualquer cliente
        cliente = Cliente.query.get_or_404(cliente_id)
        
        tipo = request.form.get('tipo')
        valor = float(request.form.get('valor'))
        descricao = request.form.get('descricao')
        
        if not tipo or not valor or not descricao:
            flash('Todos os campos são obrigatórios!', 'error')
            return redirect(url_for('gerenciar_clientes'))
        
        # Criar transação
        nova_transacao = criar_com_usuario(
            TransacaoCliente,
            cliente_id=cliente_id,
            tipo=tipo,
            valor=valor,
            descricao=descricao,
            target_user_id=cliente.user_id  # Manter na mesma conta do cliente
        )
        
        # Atualizar valor total do cliente
        if tipo == 'Recebimento':
            cliente.valor_total += valor
        else:  # Pagamento
            cliente.valor_total -= valor
        
        cliente.ultima_transacao = datetime.utcnow().date()
        
        db.session.add(nova_transacao)
        db.session.commit()
        
        flash('Transação adicionada com sucesso!', 'success')
        
    except ValueError:
        flash('Valor inválido!', 'error')
    except Exception as e:
        db.session.rollback()
        flash('Erro ao adicionar transação. Tente novamente.', 'error')
        app.logger.error(f"Erro ao adicionar transação: {str(e)}")
    
    return redirect(url_for('gerenciar_clientes'))





# COMANDO CLI REMOVIDO - APENAS ROOT E ALOIZIOTADEU


# Headers de segurança e otimização
@app.after_request
def add_security_and_cache_headers(response):
    """Adiciona headers de segurança e cache para otimizar performance e segurança"""
    
    # HEADERS DE SEGURANÇA - Aplicados a todas as responses
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://maps.googleapis.com https://*.googleapis.com https://*.gstatic.com https://*.ggpht.com; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://fonts.googleapis.com https://*.googleapis.com; font-src 'self' data: https://cdnjs.cloudflare.com https://fonts.gstatic.com https://*.gstatic.com; img-src 'self' data: https: https://*.googleapis.com https://*.gstatic.com https://*.ggpht.com blob:; connect-src 'self' https://cdn.jsdelivr.net https://maps.googleapis.com https://*.googleapis.com https://*.gstatic.com; worker-src 'self' blob:; child-src 'self' blob:;"
    
    # Headers de cache apenas para arquivos estáticos
    if request.endpoint == 'static':
        # Cache arquivos estáticos por 1 hora
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['Expires'] = datetime.utcnow() + timedelta(hours=1)
    elif request.endpoint in ['health']:
        # Health check sem cache
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    
    return response


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
