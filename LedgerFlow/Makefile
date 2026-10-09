.PHONY: up down seed test lint test-web test-api test-worker lint-web lint-api lint-worker

up:
	docker compose up --build -d

down:
	docker compose down

seed:
	docker compose exec api ledgerflow-seed

test: test-web test-api test-worker

lint: lint-web lint-api lint-worker

test-web:
	cd apps/web && npm test

test-api:
	cd services/api && uv run pytest

test-worker:
	$(MAKE) -C services/worker test

lint-web:
	cd apps/web && npm run typecheck && npm run lint && npm run format:check

lint-api:
	cd services/api && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests migrations

lint-worker:
	$(MAKE) -C services/worker lint
