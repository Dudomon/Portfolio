#!/usr/bin/env python3
"""
Script para listar todos os usuários do sistema
"""
import sys
import os

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, Usuario

def list_users():
    """Lista todos os usuários do sistema"""
    with app.app_context():
        users = Usuario.query.order_by(Usuario.id).all()

        if not users:
            print("❌ Nenhum usuário encontrado no banco de dados")
            return

        print(f"\n{'='*120}")
        print(f"📋 LISTA DE USUÁRIOS ({len(users)} encontrados)")
        print(f"{'='*120}\n")

        print(f"{'ID':<5} {'USERNAME':<20} {'NOME COMPLETO':<30} {'STATUS':<12} {'EMAIL':<30}")
        print(f"{'-'*120}")

        for user in users:
            status = user.status_aprovacao.upper()
            status_color = "✅" if status == "APROVADO" else ("⚠️" if status == "PENDENTE" else "❌")

            print(f"{user.id:<5} {user.username:<20} {(user.nome_completo or '-'):<30} {status_color} {status:<10} {(user.email or '-'):<30}")

        print(f"\n{'='*120}\n")

        # Estatísticas
        aprovados = len([u for u in users if u.status_aprovacao == 'aprovado'])
        pendentes = len([u for u in users if u.status_aprovacao == 'pendente'])
        rejeitados = len([u for u in users if u.status_aprovacao == 'rejeitado'])

        print(f"📊 ESTATÍSTICAS:")
        print(f"   ✅ Aprovados: {aprovados}")
        print(f"   ⚠️  Pendentes: {pendentes}")
        print(f"   ❌ Rejeitados: {rejeitados}")
        print(f"   📈 Total: {len(users)}\n")

if __name__ == '__main__':
    list_users()
