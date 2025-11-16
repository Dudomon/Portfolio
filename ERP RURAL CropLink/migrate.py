#!/usr/bin/env python3
"""
Script para criar/atualizar tabelas do banco de dados
"""
from app import app, db
from sqlalchemy import text

def run_migrations():
    """Executa migrações específicas que não podem ser feitas com create_all()"""

    migrations = [
        {
            'name': 'Adicionar campos talhao e umidade em movimentacao_silo',
            'sql': [
                "ALTER TABLE movimentacao_silo ADD COLUMN IF NOT EXISTS talhao VARCHAR(100);",
                "ALTER TABLE movimentacao_silo ADD COLUMN IF NOT EXISTS umidade FLOAT;"
            ]
        },
        {
            'name': 'Adicionar relacionamento talhao_id em movimentacao_silo',
            'sql': [
                "ALTER TABLE movimentacao_silo ADD COLUMN IF NOT EXISTS talhao_id INTEGER REFERENCES talhao(id);"
            ]
        },
        {
            'name': 'Adicionar campo aplicado_todos_talhoes em registro_chuva',
            'sql': [
                "ALTER TABLE registro_chuva ADD COLUMN IF NOT EXISTS aplicado_todos_talhoes BOOLEAN DEFAULT FALSE;"
            ]
        },
        {
            'name': 'Criar tabela de associação registro_chuva_talhao',
            'sql': [
                """
                CREATE TABLE IF NOT EXISTS registro_chuva_talhao (
                    registro_chuva_id INTEGER NOT NULL REFERENCES registro_chuva(id) ON DELETE CASCADE,
                    talhao_id INTEGER NOT NULL REFERENCES talhao(id) ON DELETE CASCADE,
                    PRIMARY KEY (registro_chuva_id, talhao_id)
                );
                """
            ]
        }
    ]

    for migration in migrations:
        print(f"\n🔧 Executando: {migration['name']}")
        try:
            for sql_statement in migration['sql']:
                db.session.execute(text(sql_statement))
            db.session.commit()
            print(f"   ✅ Migração concluída")
        except Exception as e:
            print(f"   ⚠️  {str(e)}")
            db.session.rollback()

if __name__ == '__main__':
    with app.app_context():
        print("🔄 Criando/atualizando tabelas do banco de dados...")

        # Criar tabelas novas
        db.create_all()
        print("✅ Tabelas base criadas/verificadas!")

        # Executar migrações específicas
        run_migrations()

        print("\n📋 Tabelas disponíveis:")
        for table_name in sorted(db.metadata.tables.keys()):
            print(f"   - {table_name}")

        print("\n✅ Migração completa!")
