"""Create a demo tenant with one user per role. Safe to run repeatedly.

    python -m ledgerflow_api.seed

The password comes from ``SEED_PASSWORD`` (default ``ledgerflow-demo``); it is meant for local
environments only.
"""

import asyncio
import os

from sqlalchemy import func, select

from ledgerflow_api.config import get_settings
from ledgerflow_api.db.models import Tenant, User
from ledgerflow_api.db.session import create_engine, create_session_factory, system_session
from ledgerflow_api.domain.roles import Role
from ledgerflow_api.security import hash_password

DEMO_TENANT_SLUG = "demo"
DEMO_USERS = (
    ("admin@demo.example.com", "Ana Admin", Role.ADMIN),
    ("reviewer@demo.example.com", "Rafael Reviewer", Role.REVIEWER),
    ("viewer@demo.example.com", "Vera Viewer", Role.VIEWER),
)


async def seed(password: str) -> None:
    engine = create_engine(get_settings())
    try:
        async with system_session(create_session_factory(engine)) as session:
            tenant = await session.scalar(select(Tenant).where(Tenant.slug == DEMO_TENANT_SLUG))
            if tenant is None:
                tenant = Tenant(name="Demo Distribuidora Ltda", slug=DEMO_TENANT_SLUG)
                session.add(tenant)
                await session.flush()

            for email, name, role in DEMO_USERS:
                exists = await session.scalar(
                    select(User.id).where(func.lower(User.email) == email.lower())
                )
                if exists is None:
                    session.add(
                        User(
                            tenant_id=tenant.id,
                            email=email,
                            name=name,
                            role=role,
                            password_hash=hash_password(password),
                        )
                    )
                    print(f"created {role.value:<8} {email}")
            await session.commit()
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(seed(os.environ.get("SEED_PASSWORD", "ledgerflow-demo")))


if __name__ == "__main__":
    main()
