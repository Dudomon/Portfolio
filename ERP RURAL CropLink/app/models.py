"""
Modelos de Dados - Sistema CropLink
==================================

Este módulo contém todos os modelos de dados SQLAlchemy para o sistema de gestão agrícola.
Implementa um sistema multi-tenant com isolamento de dados por usuário e suporte a três
níveis hierárquicos: Super Admin, Cliente (Produtor Rural) e Funcionário.

Características principais:
- Sistema hierárquico de 3 níveis de usuários
- Isolamento completo de dados por tenant (user_id)
- Gestão de planos de assinatura (basic, plus, premium)
- Sistema de aprovação de usuários
- Modelos para gestão agrícola completa

Modelos incluídos:
- Usuario: Gestão de usuários e autenticação
- ProdutorRural: Cliente (nível superior na hierarquia)
- Funcionario: Funcionários vinculados aos produtores
- Diarista: Trabalhadores diários
- Insumo/InsumoAgricola: Gestão de insumos
- Silo/Grao: Gestão de armazenamento
- Maquinario: Equipamentos agrícolas
- RegistroChuva: Monitoramento meteorológico
- Cliente/TransacaoCliente: Gestão financeira

Author: CropLink Development Team
Created: 2025-09-24
Version: 2.0.0
License: Proprietary - Fazenda Rebelato
"""

from datetime import datetime, timedelta
from sqlalchemy import Date, Time, func, Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey, CheckConstraint, Table
from sqlalchemy.orm import relationship
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from .cache import cached

def init_models(db):
    """
    Inicializa todos os modelos SQLAlchemy com a instância do banco de dados.
    
    Esta função cria todas as classes de modelo usando a instância SQLAlchemy
    fornecida, garantindo que todos os relacionamentos e restrições sejam
    configurados corretamente.
    
    Args:
        db (SQLAlchemy): Instância configurada do SQLAlchemy
        
    Returns:
        dict: Dicionário com todos os modelos criados, permitindo fácil
              importação e uso em outras partes da aplicação
              
    Example:
        models = init_models(db)
        Usuario = models['Usuario']
        Silo = models['Silo']
    """
    
    class Usuario(UserMixin, db.Model):
        """
        Modelo de usuário com sistema hierárquico e multi-tenant.
        
        Implementa um sistema de três níveis hierárquicos:
        1. Super Admin: Acesso global ao sistema (user_role='super_admin')
        2. Cliente/Produtor Rural: Proprietário dos dados (user_role='cliente')
        3. Funcionário: Vinculado a um produtor rural (user_role='funcionario')
        
        Características:
        - Autenticação via Flask-Login
        - Sistema de aprovação obrigatória
        - Planos de assinatura (basic, plus, premium)
        - Verificação de email obrigatória
        - Controle de primeiro acesso
        - Tokens de segurança para reset/verificação
        
        Campos de segurança:
        - password_hash: Senha criptografada com bcrypt
        - token_verificacao: Token único para verificação de email
        - status_aprovacao: Controle de aprovação (pendente/aprovado/rejeitado)
        - email_verificado: Flag de verificação de email
        
        Multi-tenancy:
        - Cada usuário 'cliente' é um tenant isolado
        - Funcionários são vinculados via produtor_rural_id
        - Super admins têm visão global
        """
        id = Column(Integer, primary_key=True)
        username = Column(String(80), unique=True, nullable=False)
        password_hash = Column(String(128), nullable=False)
        nome_completo = Column(String(200), nullable=False)  # Nome completo obrigatório
        email = Column(String(120), unique=True, nullable=False)  # Email obrigatório e único
        
        # Campos para controle de aprovação
        status_aprovacao = Column(String(20), nullable=False, default='pendente')  # pendente, aprovado, rejeitado
        data_cadastro = Column(DateTime, nullable=False, default=datetime.utcnow)
        data_aprovacao = Column(DateTime, nullable=True)
        aprovado_por_id = Column(Integer, ForeignKey('usuario.id'), nullable=True)
        
        # Campos para controle de primeiro acesso e verificação
        primeiro_acesso = Column(Boolean, nullable=False, default=True)
        data_ultimo_login = Column(DateTime, nullable=True)
        token_verificacao = Column(String(64), nullable=True, unique=True)
        token_expiracao = Column(DateTime, nullable=True)
        email_verificado = Column(Boolean, nullable=False, default=False)
        
        # Campos para sistema de planos por assinatura
        plano = Column(String(20), nullable=False, default='basic')
        data_expiracao = Column(Date, nullable=True)
        status_assinatura = Column(String(20), nullable=False, default='ativa')
        
        # Campo para período de teste de 5 dias
        data_inicio_teste = Column(DateTime, nullable=True, default=datetime.utcnow)
        
        # Campo para privilégios administrativos
        is_admin = Column(Boolean, nullable=False, default=False)
        
        # Campos para sistema hierárquico de 3 níveis
        user_role = Column(String(20), nullable=False, default='funcionario')  # super_admin, cliente, funcionario
        produtor_rural_id = Column(Integer, ForeignKey('produtor_rural.id'), nullable=True)  # NULL para super_admin, preenchido para funcionario
        
        # Relacionamentos
        aprovado_por = relationship('Usuario', remote_side=[id], backref='usuarios_aprovados', lazy=True)
        produtor = relationship('ProdutorRural', foreign_keys=[produtor_rural_id], backref='funcionarios', lazy=True)
        
        def tem_acesso_modulo(self, modulo):
            """
            Verifica se o usuário tem acesso ao módulo baseado no plano de assinatura.
            
            Valida se o usuário pode acessar uma funcionalidade específica
            considerando seu plano atual e status da assinatura.
            
            Args:
                modulo (str): Nome do módulo a verificar
                             Módulos disponíveis: 'silos', 'graos', 'dashboard',
                             'pulverizacao', 'caderno_campo', 'bolsa_valores'
                             
            Returns:
                bool: True se o usuário tem acesso, False caso contrário
                
            Example:
                if user.tem_acesso_modulo('pulverizacao'):
                    # Usuário pode acessar funcionalidades de pulverização
                    pass
            """
            planos_config = {
                'basic': ['silos', 'graos', 'dashboard'],
                'plus': ['silos', 'graos', 'pulverizacao', 'dashboard'],
                'premium': ['silos', 'graos', 'pulverizacao', 'caderno_campo', 'bolsa_valores', 'dashboard']
            }
            
            # Verificar se assinatura está ativa
            if self.status_assinatura != 'ativa':
                return False
                
            # Verificar se o plano ainda não expirou
            if self.data_expiracao and self.data_expiracao < datetime.now().date():
                return False
                
            return modulo in planos_config.get(self.plano, [])
            
        def esta_assinatura_valida(self):
            """
            Verifica se a assinatura do usuário está válida e ativa.

            Valida múltiplos critérios para determinar se o usuário
            pode continuar usando o sistema:
            1. Status da assinatura deve ser 'ativa'
            2. Data de expiração deve estar no futuro (se definida)
            3. Não aplica validação para admins (acesso irrestrito)

            Returns:
                bool: True se assinatura válida, False se expirada/inativa

            Note:
                Administradores sempre retornam True independente da assinatura
            """
            if self.status_assinatura != 'ativa':
                return False
            # Para contas pagas, data_expiracao deve estar definida e ser futura
            if not self.data_expiracao:
                return False  # Sem data de expiração = sem assinatura paga válida
            try:
                if self.data_expiracao < datetime.now().date():
                    return False
            except (TypeError, AttributeError):
                # Em caso de erro na comparação de datas
                return False
            return True
            
        def esta_no_periodo_teste(self):
            """Verifica se o usuário ainda está no período de teste de 5 dias exatos"""
            if not self.data_inicio_teste:
                # Se não tem data de início, considerar que está fora do período de teste
                # Isso evita erro 500 em usuários antigos
                return False
            try:
                tempo_desde_inicio = datetime.utcnow() - self.data_inicio_teste
                return tempo_desde_inicio <= timedelta(days=5)
            except (TypeError, AttributeError):
                # Em caso de erro (data inválida), retornar False
                return False
            
        def periodo_teste_expirou(self):
            """Verifica se o período de teste de 5 dias exatos expirou"""
            if not self.data_inicio_teste:
                # Se não tem data de início, considerar que não expirou
                return False
            try:
                tempo_desde_inicio = datetime.utcnow() - self.data_inicio_teste
                return tempo_desde_inicio > timedelta(days=5)
            except (TypeError, AttributeError):
                # Em caso de erro, considerar que não expirou
                return False
            
        def esta_aprovado(self):
            """Verifica se o usuário foi aprovado pelo administrador"""
            return self.status_aprovacao == 'aprovado'
            
        def pode_fazer_login(self):
            """Verifica se o usuário pode fazer login (aprovado, ativo, email verificado e dentro do período de teste)"""
            try:
                import logging
                logger = logging.getLogger(__name__)

                logger.info(f"📋 pode_fazer_login() para {self.username}")
                logger.info(f"   is_admin: {self.is_admin}")
                logger.info(f"   status_assinatura: {self.status_assinatura}")
                logger.info(f"   email_verificado: {self.email_verificado}")
                logger.info(f"   status_aprovacao: {self.status_aprovacao}")

                # Administradores (super_admin ou is_admin) sempre podem fazer login sem verificar período de teste
                if self.is_super_admin() or self.is_admin:
                    resultado = (self.esta_aprovado() and
                            self.status_assinatura == 'ativa' and
                            self.email_verificado)
                    logger.info(f"   Admin: retornando {resultado}")
                    return resultado

                # Para outros usuários, verificar também o período de teste
                aprovado = self.esta_aprovado()
                assinatura_ativa = self.status_assinatura == 'ativa'
                email_ok = self.email_verificado

                logger.info(f"   aprovado: {aprovado}")
                logger.info(f"   assinatura_ativa: {assinatura_ativa}")
                logger.info(f"   email_ok: {email_ok}")

                no_periodo = self.esta_no_periodo_teste()
                logger.info(f"   esta_no_periodo_teste(): {no_periodo}")

                assinatura_valida = self.esta_assinatura_valida()
                logger.info(f"   esta_assinatura_valida(): {assinatura_valida}")

                resultado = (aprovado and assinatura_ativa and email_ok and (no_periodo or assinatura_valida))
                logger.info(f"   RESULTADO FINAL: {resultado}")
                return resultado

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"🔥 EXCEÇÃO em pode_fazer_login(): {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                raise
            
        def precisa_trocar_senha(self):
            """Verifica se é o primeiro acesso e precisa trocar a senha"""
            return self.primeiro_acesso
            
        def marcar_primeiro_acesso_concluido(self):
            """Marca que o primeiro acesso foi concluído"""
            self.primeiro_acesso = False
            self.data_ultimo_login = datetime.utcnow()
            
        def gerar_token_verificacao(self):
            """Gera um token único para verificação de email"""
            import secrets
            self.token_verificacao = secrets.token_urlsafe(32)
            self.token_expiracao = datetime.utcnow() + timedelta(hours=24)
            return self.token_verificacao
            
        def token_valido(self):
            """Verifica se o token de verificação ainda é válido"""
            return (self.token_verificacao and 
                    self.token_expiracao and 
                    datetime.utcnow() < self.token_expiracao)
                    
        def marcar_email_verificado(self):
            """Marca o email como verificado e limpa o token"""
            self.email_verificado = True
            self.token_verificacao = None
            self.token_expiracao = None
            
        def aprovar(self, admin_user_id):
            """Aprova o usuário"""
            self.status_aprovacao = 'aprovado'
            self.data_aprovacao = datetime.utcnow()
            self.aprovado_por_id = admin_user_id
            
        def rejeitar(self, admin_user_id):
            """Rejeita o usuário"""
            self.status_aprovacao = 'rejeitado'
            self.data_aprovacao = datetime.utcnow()
            self.aprovado_por_id = admin_user_id
            
        # Métodos para sistema hierárquico
        def is_super_admin(self):
            """Verifica se é super administrador (dono da plataforma)"""
            return self.user_role == 'super_admin'
            
        def is_cliente(self):
            """Verifica se é um cliente (produtor rural)"""
            return self.user_role == 'cliente'
            
        def is_funcionario(self):
            """Verifica se é um funcionário"""
            return self.user_role == 'funcionario'
            
        def pode_gerenciar_clientes(self):
            """Verifica se pode gerenciar clientes (só super_admin)"""
            return self.is_super_admin()
            
        def pode_gerenciar_funcionarios(self):
            """Verifica se pode gerenciar funcionários da sua empresa"""
            return self.is_cliente() or self.is_super_admin()
            
        def get_produtor_contexto(self):
            """Retorna o contexto do produtor rural para isolamento de dados"""
            if self.is_super_admin():
                return None  # Super admin vê todos os dados
            elif self.is_cliente():
                # Para cliente (produtor rural), busca seu próprio ID na tabela produtor_rural
                from sqlalchemy import text
                result = db.session.execute(
                    text("SELECT id FROM produtor_rural WHERE id = (SELECT produtor_rural_id FROM usuario WHERE id = :user_id AND user_role = 'cliente')"),
                    {'user_id': self.id}
                ).fetchone()
                return result[0] if result else self.produtor_rural_id
            elif self.is_funcionario():
                return self.produtor_rural_id  # Funcionário pertence a um produtor rural
            return None

    class ProdutorRural(db.Model):
        """Modelo para clientes (produtores rurais) do sistema"""
        __tablename__ = 'produtor_rural'
        id = Column(Integer, primary_key=True)
        nome_fazenda = Column(String(200), nullable=False)
        proprietario_nome = Column(String(200), nullable=False)  
        cpf_cnpj = Column(String(18), unique=True, nullable=False)
        telefone = Column(String(20), nullable=True)
        email = Column(String(120), nullable=False)
        endereco = Column(Text, nullable=True)
        cidade = Column(String(100), nullable=True)
        estado = Column(String(2), nullable=True)  
        cep = Column(String(10), nullable=True)
        
        # Dados operacionais
        area_total_hectares = Column(Float, nullable=True)
        culturas_principais = Column(Text, nullable=True)  # JSON ou texto separado por vírgula
        
        # Dados comerciais
        plano = Column(String(20), nullable=False, default='basic')
        data_contratacao = Column(DateTime, nullable=False, default=datetime.utcnow)
        data_expiracao = Column(Date, nullable=True)
        status = Column(String(20), nullable=False, default='ativo')  # ativo, suspenso, cancelado
        valor_mensalidade = Column(Float, nullable=True)
        
        # Controle
        criado_em = Column(DateTime, nullable=False, default=datetime.utcnow)
        criado_por_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        
        # Relacionamentos
        criado_por = relationship('Usuario', foreign_keys=[criado_por_id], lazy=True)
        # funcionarios definido no backref do Usuario
        
        def esta_ativo(self):
            """Verifica se o cliente está ativo e com contrato válido"""
            if self.status != 'ativo':
                return False
            if self.data_expiracao and self.data_expiracao < datetime.now().date():
                return False
            return True
            
        def dias_ate_vencimento(self):
            """Calcula quantos dias restam até o vencimento do contrato"""
            if not self.data_expiracao:
                return None
            delta = self.data_expiracao - datetime.now().date()
            return delta.days if delta.days >= 0 else 0

    class Insumo(db.Model):
        """Modelo para insumos gerais"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='insumo_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome = Column(String(100), nullable=False)
        quantidade = Column(Float, nullable=False)
        unidade = Column(String(20), nullable=False)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        movimentacoes = relationship('MovimentacaoInsumo', backref='insumo', lazy=True, cascade="all, delete-orphan")
        proprietario = relationship('Usuario', backref='insumos', lazy=True)
        
        def prever_duracao_estoque(self, periodo_dias=30):
            """Prevê quantos dias o estoque atual durará baseado no consumo médio"""
            if self.quantidade <= 0:
                return 0
            
            data_inicio_analise = datetime.utcnow() - timedelta(days=periodo_dias)
            consumo_total = db.session.query(func.sum(MovimentacaoInsumo.quantidade)).filter(
                MovimentacaoInsumo.insumo_id == self.id,
                MovimentacaoInsumo.tipo == 'Saída',
                MovimentacaoInsumo.data >= data_inicio_analise,
                MovimentacaoInsumo.user_id == self.user_id
            ).scalar() or 0.0
            
            if consumo_total == 0:
                return -1  # Consumo zero indica duração indefinida
                
            consumo_medio_diario = consumo_total / periodo_dias
            if consumo_medio_diario <= 0:
                return -1
                
            dias_restantes = self.quantidade / consumo_medio_diario
            return int(dias_restantes)

    class MovimentacaoInsumo(db.Model):
        """Modelo para movimentações de insumos"""
        __table_args__ = (
            # Constraint para garantir isolamento multi-tenant
            # Nota: Este constraint será validado na aplicação devido à complexidade do subquery
            CheckConstraint('user_id IS NOT NULL', name='movimentacao_insumo_user_not_null'),
        )
        
        id = Column(Integer, primary_key=True)
        tipo = Column(String(10), nullable=False)  # 'Entrada' ou 'Saída'
        quantidade = Column(Float, nullable=False)
        data = Column(DateTime, nullable=False, default=datetime.utcnow)
        observacao = Column(Text, nullable=True)
        insumo_id = Column(Integer, ForeignKey('insumo.id'), nullable=False)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        proprietario = relationship('Usuario', backref='movimentacoes_insumo', lazy=True)

    class InsumoAgricola(db.Model):
        """Modelo para insumos agrícolas específicos"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='insumo_agricola_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome = Column(String(100), nullable=False)
        quantidade = Column(Float, nullable=False)
        unidade = Column(String(50), nullable=False)
        observacao = Column(Text, nullable=True)
        categoria = Column(String(50), nullable=False)  # Herbicida, Fertilizante, etc.
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        movimentacoes = relationship('MovimentacaoInsumoAgricola', backref='insumo_agricola', lazy=True, cascade="all, delete-orphan")
        proprietario = relationship('Usuario', backref='insumos_agricolas', lazy=True)

    class MovimentacaoInsumoAgricola(db.Model):
        """Modelo para movimentações de insumos agrícolas com detalhes de aplicação"""
        __table_args__ = (
            # Constraint para garantir isolamento multi-tenant
            CheckConstraint('user_id IS NOT NULL', name='movimentacao_insumo_agricola_user_not_null'),
        )
        
        id = Column(Integer, primary_key=True)
        tipo = Column(String(10), nullable=False)  # 'Entrada' ou 'Saída'
        quantidade = Column(Float, nullable=False)
        data = Column(DateTime, nullable=False, default=datetime.utcnow)
        observacao = Column(Text, nullable=True)
        talhao = Column(String(100), nullable=True)
        condicao_aplicacao = Column(String(50), nullable=True)
        dose_aplicada = Column(Float, nullable=True)
        unidade_dose = Column(String(50), nullable=True)
        insumo_agricola_id = Column(Integer, ForeignKey('insumo_agricola.id'), nullable=False)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        proprietario = relationship('Usuario', backref='movimentacoes_insumo_agricola', lazy=True)

    class Maquinario(db.Model):
        """Modelo para máquinas e equipamentos"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='maquinario_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome = Column(String(100), nullable=False)
        marca = Column(String(50))
        modelo = Column(String(50))
        ano = Column(Integer)
        status = Column(String(50), default='Operacional')
        tipo_oleo = Column(String(50))
        filtro_oleo = Column(String(50))
        filtro_ar = Column(String(50))
        filtro_combustivel = Column(String(50))
        # Campos de datas das últimas trocas/manutenções
        data_ultima_troca_oleo = Column(Date)
        data_ultima_troca_filtro_oleo = Column(Date)
        data_ultima_troca_filtro_ar = Column(Date)
        data_ultima_troca_filtro_combustivel = Column(Date)
        horas_de_uso = Column(Float)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        proprietario = relationship('Usuario', backref='maquinarios', lazy=True)

    class Funcionario(db.Model):
        """Modelo para funcionários fixos"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='funcionario_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome_completo = Column(String(150), nullable=False)
        cpf = Column(String(14))
        telefone = Column(String(20))
        cargo = Column(String(100))
        data_admissao = Column(Date)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        proprietario = relationship('Usuario', backref='funcionarios', lazy=True)

    class Diarista(db.Model):
        """Modelo para trabalhadores diaristas"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='diarista_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome_completo = Column(String(150), nullable=False)
        cpf = Column(String(14))
        telefone = Column(String(20))
        valor_diaria = Column(Float)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        registros = relationship('RegistroDiaria', backref='diarista', lazy=True, cascade="all, delete-orphan")
        proprietario = relationship('Usuario', backref='diaristas', lazy=True)

    class RegistroDiaria(db.Model):
        """Modelo para registro de trabalho de diaristas"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='registro_diaria_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        data = Column(Date, nullable=False, default=datetime.utcnow)
        hora_entrada = Column(Time)
        hora_saida = Column(Time)
        descricao_trabalho = Column(Text)
        observacoes = Column(Text)
        diarista_id = Column(Integer, ForeignKey('diarista.id'), nullable=False)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        proprietario = relationship('Usuario', backref='registros_diaria', lazy=True)

    class Silo(db.Model):
        """Modelo para silos de armazenamento"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='silo_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome = Column(String(50), nullable=False)
        capacidade_kg = Column(Float)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        movimentacoes = relationship('MovimentacaoSilo', backref='silo', lazy=True, cascade="all, delete-orphan")
        proprietario = relationship('Usuario', backref='silos', lazy=True)
        
        def get_estoque_por_grao(self, grao_id):
            """Calcula o estoque atual de um grão específico neste silo"""
            entradas = db.session.query(func.sum(MovimentacaoSilo.quantidade_kg)).filter(
                MovimentacaoSilo.silo_id == self.id,
                MovimentacaoSilo.grao_id == grao_id,
                MovimentacaoSilo.tipo_movimentacao == 'Entrada',
                MovimentacaoSilo.user_id == self.user_id
            ).scalar() or 0.0
            
            saidas = db.session.query(func.sum(MovimentacaoSilo.quantidade_kg)).filter(
                MovimentacaoSilo.silo_id == self.id,
                MovimentacaoSilo.grao_id == grao_id,
                MovimentacaoSilo.tipo_movimentacao == 'Saída',
                MovimentacaoSilo.user_id == self.user_id
            ).scalar() or 0.0
            
            return entradas - saidas
            
        @cached('silo_occupancy', key_func=lambda self: f"silo_stock:{self.id}")
        def get_estoque_total(self):
            """Calcula o estoque total de todos os grãos neste silo - CACHE: TTL 30s"""
            entradas = db.session.query(func.sum(MovimentacaoSilo.quantidade_kg)).filter(
                MovimentacaoSilo.silo_id == self.id,
                MovimentacaoSilo.tipo_movimentacao == 'Entrada',
                MovimentacaoSilo.user_id == self.user_id
            ).scalar() or 0.0
            
            saidas = db.session.query(func.sum(MovimentacaoSilo.quantidade_kg)).filter(
                MovimentacaoSilo.silo_id == self.id,
                MovimentacaoSilo.tipo_movimentacao == 'Saída',
                MovimentacaoSilo.user_id == self.user_id
            ).scalar() or 0.0
            
            return entradas - saidas
            
        @cached('silo_occupancy', key_func=lambda self: f"silo_percent:{self.id}")
        def get_percentual_ocupacao(self):
            """Calcula o percentual de ocupação do silo - CACHE: TTL 30s"""
            if not self.capacidade_kg or self.capacidade_kg <= 0:
                return 0
            estoque_total = self.get_estoque_total()
            return (estoque_total / self.capacidade_kg) * 100
            
        @cached('silo_occupancy', key_func=lambda self: f"silo_capacity:{self.id}")
        def get_capacidade_disponivel(self):
            """Calcula a capacidade disponível do silo - CACHE: TTL 30s"""
            if not self.capacidade_kg:
                return 0
            return self.capacidade_kg - self.get_estoque_total()
            
        def calcular_sacas(self, peso_kg):
            """Calcula o número de sacas baseado no peso (1 saca = 60kg)"""
            return peso_kg / 60.0

    class Grao(db.Model):
        """Modelo para tipos de grãos"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='grao_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome = Column(String(50), nullable=False)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        movimentacoes = relationship('MovimentacaoSilo', backref='grao', lazy=True, cascade="all, delete-orphan")
        proprietario = relationship('Usuario', backref='graos', lazy=True)

    class MovimentacaoSilo(db.Model):
        """Modelo para movimentações de grãos em silos com controle de transporte"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='movimentacao_silo_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        tipo_movimentacao = Column(String(10), nullable=False)  # 'Entrada' ou 'Saída'
        quantidade_kg = Column(Float, nullable=False)
        data_movimentacao = Column(DateTime, nullable=False, default=datetime.utcnow)
        observacao = Column(Text, nullable=True)
        silo_id = Column(Integer, ForeignKey('silo.id'), nullable=False)
        grao_id = Column(Integer, ForeignKey('grao.id'), nullable=False)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)

        # Campos de controle de transporte
        placa_caminhao = Column(String(20), nullable=True)
        nome_motorista = Column(String(100), nullable=True)
        peso_entrada_kg = Column(Float, nullable=True)
        peso_saida_kg = Column(Float, nullable=True)
        peso_liquido_kg = Column(Float, nullable=True)

        # Campos de origem e qualidade
        talhao = Column(String(100), nullable=True)  # Talhão de origem dos grãos (texto livre - DEPRECATED)
        talhao_id = Column(Integer, ForeignKey('talhao.id'), nullable=True)  # Talhão cadastrado
        umidade = Column(Float, nullable=True)  # Percentual de umidade (%)

        proprietario = relationship('Usuario', backref='movimentacoes_silo', lazy=True)
        talhao_origem = relationship('Talhao', backref='movimentacoes', lazy=True)
        
        def calcular_peso_liquido(self):
            """Calcula o peso líquido automaticamente (tara) baseado no tipo de movimentação"""
            if self.peso_entrada_kg and self.peso_saida_kg:
                if self.tipo_movimentacao == 'Entrada':
                    # ENTRADA NO SILO:
                    # Peso Entrada = Caminhão CHEIO
                    # Peso Saída = Caminhão VAZIO
                    # Peso Líquido = Entrada - Saída
                    return self.peso_entrada_kg - self.peso_saida_kg
                else:  # 'Saída'
                    # SAÍDA DO SILO:
                    # Peso Entrada = Caminhão VAZIO
                    # Peso Saída = Caminhão CHEIO
                    # Peso Líquido = Saída - Entrada
                    return self.peso_saida_kg - self.peso_entrada_kg
            return None
            
        def get_peso_final(self):
            """Retorna o peso líquido ou a quantidade original"""
            return self.peso_liquido_kg or self.quantidade_kg

    # Tabela de associação para relação many-to-many entre RegistroChuva e Talhao
    registro_chuva_talhao = Table('registro_chuva_talhao', db.Model.metadata,
        Column('registro_chuva_id', Integer, ForeignKey('registro_chuva.id'), primary_key=True),
        Column('talhao_id', Integer, ForeignKey('talhao.id'), primary_key=True)
    )

    class RegistroChuva(db.Model):
        """Modelo para registro de chuvas vinculado a talhões"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='registro_chuva_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        data = Column(Date, nullable=False, default=datetime.utcnow)
        quantidade_mm = Column(Float, nullable=False)
        observacao = Column(Text, nullable=True)
        aplicado_todos_talhoes = Column(Boolean, nullable=False, default=False)  # Se True, aplica a todos os talhões do usuário
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        proprietario = relationship('Usuario', backref='registros_chuva', lazy=True)
        talhoes = relationship('Talhao', secondary=registro_chuva_talhao, backref='registros_chuva', lazy=True)

    class Talhao(db.Model):
        """Modelo para talhões/áreas agrícolas desenhadas no mapa"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='talhao_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome = Column(String(100), nullable=False)
        area_hectares = Column(Float, nullable=True)  # Área em hectares
        area_alqueires = Column(Float, nullable=True)  # Área em alqueires paulista (2.42 ha)
        coordenadas = Column(Text, nullable=False)  # JSON com array de lat/lng: [{"lat": -23.5, "lng": -46.6}, ...]
        cor = Column(String(7), nullable=True, default='#FFD700')  # Cor do polígono no mapa (hex)
        observacao = Column(Text, nullable=True)
        data_criacao = Column(DateTime, nullable=False, default=datetime.utcnow)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        proprietario = relationship('Usuario', backref='talhoes', lazy=True)

        def get_area_display(self):
            """Retorna área formatada em hectares"""
            if self.area_hectares:
                return f"{self.area_hectares:.2f} ha"
            elif self.area_alqueires:
                # Converter alqueires para hectares (1 alqueire paulista = 2.42 ha)
                ha = self.area_alqueires * 2.42
                return f"{ha:.2f} ha"
            return "N/A"

    class Cliente(db.Model):
        """Modelo para clientes e suas informações financeiras"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='cliente_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome = Column(String(150), nullable=False)
        empresa = Column(String(200), nullable=True)
        categoria = Column(String(50), nullable=False)  # Fornecedor, Comprador, Prestador de Serviço
        documento = Column(String(20), nullable=True)  # CPF/CNPJ
        telefone = Column(String(20), nullable=True)
        email = Column(String(120), nullable=True)
        endereco = Column(Text, nullable=True)
        status = Column(String(20), nullable=False, default='Ativo')  # Ativo, Inativo, Pendente
        valor_total = Column(Float, nullable=False, default=0.0)
        ultima_transacao = Column(Date, nullable=True)
        observacoes = Column(Text, nullable=True)
        data_cadastro = Column(DateTime, nullable=False, default=datetime.utcnow)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        proprietario = relationship('Usuario', backref='clientes', lazy=True)
        transacoes = relationship('TransacaoCliente', backref='cliente', lazy=True, cascade="all, delete-orphan")

    class TransacaoCliente(db.Model):
        """Modelo para transações financeiras com clientes"""
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='transacao_cliente_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        cliente_id = Column(Integer, ForeignKey('cliente.id'), nullable=False)
        tipo = Column(String(20), nullable=False)  # Recebimento, Pagamento
        valor = Column(Float, nullable=False)
        descricao = Column(Text, nullable=False)
        data_transacao = Column(DateTime, nullable=False, default=datetime.utcnow)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        proprietario = relationship('Usuario', backref='transacoes_clientes', lazy=True)

    class AplicacaoInsumo(db.Model):
        """Modelo para registrar aplicações de insumos agrícolas"""
        __table_args__ = (
            # Constraint para garantir isolamento multi-tenant
            CheckConstraint('user_id IS NOT NULL', name='aplicacao_insumo_user_not_null'),
        )
        
        id = Column(Integer, primary_key=True)
        insumo_agricola_id = Column(Integer, ForeignKey('insumo_agricola.id'), nullable=False)
        quantidade_aplicada = Column(Float, nullable=False)
        data_aplicacao = Column(DateTime, nullable=False, default=datetime.utcnow)
        talhao = Column(String(100), nullable=True)
        observacao = Column(Text, nullable=True)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        
        # Relacionamentos
        insumo = relationship('InsumoAgricola', backref='aplicacoes', lazy=True)
        proprietario = relationship('Usuario', backref='aplicacoes_insumos', lazy=True)

    class ContasPagar(db.Model):
        """
        Modelo para gestão de contas a pagar.
        
        Gerencia todas as obrigações financeiras da propriedade rural,
        incluindo fornecedores, valores, vencimentos e controle de pagamentos.
        
        Características:
        - Multi-tenant com isolamento por user_id
        - Controle de status (pendente, pago, vencido, cancelado)
        - Histórico de pagamentos
        - Categorização de despesas
        - Suporte a parcelas e recorrência
        """
        __tablename__ = 'contas_pagar'
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='contas_pagar_user_not_null'),
            CheckConstraint('valor > 0', name='contas_pagar_valor_positivo'),
        )
        
        id = Column(Integer, primary_key=True)
        descricao = Column(String(200), nullable=False)
        fornecedor = Column(String(200), nullable=False)
        categoria = Column(String(100), nullable=False)
        valor = Column(Float, nullable=False)
        data_emissao = Column(Date, nullable=False, default=datetime.utcnow)
        data_vencimento = Column(Date, nullable=False)
        data_pagamento = Column(Date, nullable=True)
        status = Column(String(20), nullable=False, default='pendente')
        forma_pagamento = Column(String(50), nullable=True)
        numero_documento = Column(String(100), nullable=True)
        observacoes = Column(Text, nullable=True)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        data_criacao = Column(DateTime, nullable=False, default=datetime.utcnow)
        data_atualizacao = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Relacionamentos
        proprietario = relationship('Usuario', backref='contas_pagar', lazy=True)
        
        def esta_vencida(self):
            """Verifica se a conta está vencida."""
            if self.status == 'pendente' and self.data_vencimento < datetime.now().date():
                return True
            return False
        
        def dias_ate_vencimento(self):
            """Calcula quantos dias faltam para o vencimento."""
            if self.status == 'pendente':
                delta = self.data_vencimento - datetime.now().date()
                return delta.days
            return None

    class ContasReceber(db.Model):
        """
        Modelo para gestão de contas a receber.

        Gerencia todas as receitas da propriedade rural,
        incluindo clientes, valores, vencimentos e controle de recebimentos.

        Características:
        - Multi-tenant com isolamento por user_id
        - Controle de status (pendente, recebido, vencido, cancelado)
        - Histórico de recebimentos
        - Categorização de receitas
        - Suporte a parcelas e recorrência
        """
        __tablename__ = 'contas_receber'
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='contas_receber_user_not_null'),
            CheckConstraint('valor > 0', name='contas_receber_valor_positivo'),
        )

        id = Column(Integer, primary_key=True)
        descricao = Column(String(200), nullable=False)
        cliente = Column(String(200), nullable=False)
        categoria = Column(String(100), nullable=False)
        valor = Column(Float, nullable=False)
        data_emissao = Column(Date, nullable=False, default=datetime.utcnow)
        data_vencimento = Column(Date, nullable=False)
        data_recebimento = Column(Date, nullable=True)
        status = Column(String(20), nullable=False, default='pendente')
        forma_recebimento = Column(String(50), nullable=True)
        numero_documento = Column(String(100), nullable=True)
        observacoes = Column(Text, nullable=True)
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        data_criacao = Column(DateTime, nullable=False, default=datetime.utcnow)
        data_atualizacao = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

        # Relacionamentos
        proprietario = relationship('Usuario', backref='contas_receber', lazy=True)

        def esta_vencida(self):
            """Verifica se a conta está vencida."""
            if self.status == 'pendente' and self.data_vencimento < datetime.now().date():
                return True
            return False

        def dias_ate_vencimento(self):
            """Calcula quantos dias faltam para o vencimento."""
            if self.status == 'pendente':
                delta = self.data_vencimento - datetime.now().date()
                return delta.days
            return None

    class Fornecedor(db.Model):
        """
        Modelo para gestão de fornecedores.

        Gerencia informações de fornecedores, prestadores de serviço e parceiros comerciais.

        Características:
        - Multi-tenant com isolamento por user_id
        - Informações completas de contato
        - Categorização por tipo de fornecimento
        - Controle de status (ativo/inativo)
        - Histórico de relacionamento
        """
        __tablename__ = 'fornecedor'
        __table_args__ = (
            CheckConstraint('user_id IS NOT NULL', name='fornecedor_user_not_null'),
        )

        id = Column(Integer, primary_key=True)
        nome = Column(String(200), nullable=False)
        nome_fantasia = Column(String(200), nullable=True)
        cnpj_cpf = Column(String(18), nullable=True)
        categoria = Column(String(100), nullable=False)

        # Contato
        contato_nome = Column(String(200), nullable=True)
        telefone = Column(String(20), nullable=True)
        email = Column(String(120), nullable=True)

        # Endereço
        endereco = Column(String(300), nullable=True)
        cidade = Column(String(100), nullable=True)
        estado = Column(String(2), nullable=True)
        cep = Column(String(10), nullable=True)

        # Informações adicionais
        status = Column(String(20), nullable=False, default='ativo')
        observacoes = Column(Text, nullable=True)

        # Multi-tenant
        user_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
        data_criacao = Column(DateTime, nullable=False, default=datetime.utcnow)
        data_atualizacao = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

        # Relacionamentos
        proprietario = relationship('Usuario', backref='fornecedores', lazy=True)

    # Retorna todos os modelos para uso global
    return {
        'Usuario': Usuario,
        'ProdutorRural': ProdutorRural,
        'Insumo': Insumo,
        'MovimentacaoInsumo': MovimentacaoInsumo,
        'InsumoAgricola': InsumoAgricola,
        'MovimentacaoInsumoAgricola': MovimentacaoInsumoAgricola,
        'AplicacaoInsumo': AplicacaoInsumo,
        'Maquinario': Maquinario,
        'Funcionario': Funcionario,
        'Diarista': Diarista,
        'RegistroDiaria': RegistroDiaria,
        'Silo': Silo,
        'Grao': Grao,
        'MovimentacaoSilo': MovimentacaoSilo,
        'RegistroChuva': RegistroChuva,
        'Talhao': Talhao,
        'Cliente': Cliente,
        'TransacaoCliente': TransacaoCliente,
        'ContasPagar': ContasPagar,
        'ContasReceber': ContasReceber,
        'Fornecedor': Fornecedor
    }