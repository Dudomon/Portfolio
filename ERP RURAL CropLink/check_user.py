#!/usr/bin/env python3
"""
Script de diagnóstico para verificar status de usuário
"""
import sys
import os
from datetime import datetime

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, Usuario

def check_user(username):
    """Verifica status completo de um usuário"""
    with app.app_context():
        user = Usuario.query.filter_by(username=username).first()

        if not user:
            print(f"❌ Usuário '{username}' NÃO ENCONTRADO no banco de dados")
            return

        print(f"\n{'='*60}")
        print(f"📋 DIAGNÓSTICO DO USUÁRIO: {username}")
        print(f"{'='*60}\n")

        # Informações básicas
        print(f"✅ ID: {user.id}")
        print(f"✅ Username: {user.username}")
        print(f"✅ Nome completo: {user.nome_completo or '(não definido)'}")
        print(f"✅ Email: {user.email}")
        print(f"✅ Data cadastro: {user.data_cadastro}")

        print(f"\n{'─'*60}")
        print("🔐 STATUS DE ACESSO:")
        print(f"{'─'*60}\n")

        # Status de aprovação
        status_icon = "✅" if user.status_aprovacao == 'aprovado' else "❌"
        print(f"{status_icon} Status aprovação: {user.status_aprovacao.upper()}")

        if user.status_aprovacao == 'pendente':
            print("   ⚠️  PROBLEMA: Usuário ainda está PENDENTE de aprovação")
            print("   💡 SOLUÇÃO: Um administrador precisa aprovar este usuário")

        if user.status_aprovacao == 'rejeitado':
            print("   ❌ PROBLEMA: Usuário foi REJEITADO")
            print("   💡 SOLUÇÃO: Reativar o usuário no painel administrativo")

        # Email verificado
        email_icon = "✅" if user.email_verificado else "⚠️"
        print(f"{email_icon} Email verificado: {user.email_verificado}")

        # Primeiro acesso
        primeiro_icon = "⚠️" if user.primeiro_acesso else "✅"
        print(f"{primeiro_icon} Primeiro acesso: {user.primeiro_acesso}")

        if user.primeiro_acesso:
            print("   ℹ️  Usuário precisará trocar senha no primeiro login")

        print(f"\n{'─'*60}")
        print("📊 PLANO E PERÍODO:")
        print(f"{'─'*60}\n")

        # Plano
        print(f"💎 Plano: {user.plano or '(não definido)'}")
        print(f"📅 Status assinatura: {user.status_assinatura or 'ativo'}")

        # Período de teste
        if user.data_inicio_teste:
            print(f"🧪 Início teste: {user.data_inicio_teste}")
            dias_teste = (datetime.utcnow() - user.data_inicio_teste).days
            print(f"📆 Dias de teste usados: {dias_teste}")

            if user.data_expiracao:
                print(f"⏰ Data expiração: {user.data_expiracao}")

                if datetime.utcnow() > user.data_expiracao:
                    print("   ❌ PROBLEMA: Período de teste EXPIRADO")
                    print("   💡 SOLUÇÃO: Estender período de teste ou ativar plano pago")
                else:
                    dias_restantes = (user.data_expiracao - datetime.utcnow()).days
                    print(f"   ✅ Dias restantes: {dias_restantes}")

        # Tipo de usuário
        print(f"\n{'─'*60}")
        print("👤 TIPO DE USUÁRIO:")
        print(f"{'─'*60}\n")

        print(f"🔑 Is Admin: {user.is_admin}")
        print(f"⭐ User Role: {user.user_role or 'cliente'}")

        if user.produtor_rural_id:
            print(f"🔗 Vinculado ao produtor ID: {user.produtor_rural_id}")

        # Verificações finais
        print(f"\n{'─'*60}")
        print("🔍 VERIFICAÇÕES DE LOGIN:")
        print(f"{'─'*60}\n")

        pode_logar = True
        problemas = []

        if not user.esta_aprovado():
            pode_logar = False
            problemas.append("Usuário não está APROVADO")

        if user.data_expiracao and datetime.utcnow() > user.data_expiracao and not user.is_admin:
            pode_logar = False
            problemas.append("Período de teste EXPIRADO")

        if pode_logar:
            print("✅ Usuário PODE fazer login normalmente")
        else:
            print("❌ Usuário NÃO PODE fazer login")
            print("\n🔴 PROBLEMAS ENCONTRADOS:")
            for i, problema in enumerate(problemas, 1):
                print(f"   {i}. {problema}")

        print(f"\n{'='*60}\n")

        return user

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Uso: python check_user.py <username>")
        print("\nExemplo: python check_user.py joao")
        sys.exit(1)

    username = sys.argv[1]
    check_user(username)
