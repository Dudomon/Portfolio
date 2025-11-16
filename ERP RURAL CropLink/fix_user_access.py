#!/usr/bin/env python3
"""
Script para corrigir acesso de usuário
"""
import sys
import os
from datetime import datetime, timedelta

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, Usuario

def fix_user_access(username):
    """Corrige problemas de acesso do usuário"""
    with app.app_context():
        user = Usuario.query.filter_by(username=username).first()

        if not user:
            print(f"❌ Usuário '{username}' não encontrado")
            return

        print(f"\n{'='*60}")
        print(f"🔧 CORRIGINDO ACESSO: {username}")
        print(f"{'='*60}\n")

        alteracoes = []

        # 1. Aprovar usuário se estiver pendente/rejeitado
        if user.status_aprovacao != 'aprovado':
            print(f"✓ Aprovando usuário (era: {user.status_aprovacao})...")
            user.status_aprovacao = 'aprovado'
            user.data_aprovacao = datetime.utcnow()
            alteracoes.append("Status alterado para APROVADO")

        # 2. Verificar email
        if not user.email_verificado:
            print("✓ Marcando email como verificado...")
            user.email_verificado = True
            alteracoes.append("Email marcado como VERIFICADO")

        # 3. Estender período de teste se expirado
        if user.data_expiracao and datetime.utcnow() > user.data_expiracao:
            print("✓ Período expirado - estendendo por mais 30 dias...")
            user.data_expiracao = datetime.utcnow() + timedelta(days=30)
            alteracoes.append(f"Período estendido até {user.data_expiracao.strftime('%d/%m/%Y')}")

        # 4. Garantir status de assinatura ativo
        if user.status_assinatura != 'ativo':
            print("✓ Ativando assinatura...")
            user.status_assinatura = 'ativo'
            alteracoes.append("Status da assinatura: ATIVO")

        if alteracoes:
            db.session.commit()
            print(f"\n{'─'*60}")
            print("✅ ALTERAÇÕES REALIZADAS:")
            print(f"{'─'*60}\n")
            for i, alteracao in enumerate(alteracoes, 1):
                print(f"   {i}. {alteracao}")

            print(f"\n{'='*60}")
            print(f"✅ Usuário '{username}' pode fazer login agora!")
            print(f"{'='*60}\n")
        else:
            print("ℹ️  Nenhuma alteração necessária - usuário já está OK")

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Uso: python fix_user_access.py <username>")
        print("\nExemplo: python fix_user_access.py joao")
        print("\n⚠️  Este script irá:")
        print("   - Aprovar o usuário")
        print("   - Verificar o email")
        print("   - Estender período de teste se expirado")
        print("   - Ativar a assinatura")
        sys.exit(1)

    username = sys.argv[1]

    resposta = input(f"\n⚠️  Tem certeza que deseja corrigir acesso de '{username}'? (s/N): ")

    if resposta.lower() == 's':
        fix_user_access(username)
    else:
        print("❌ Operação cancelada")
