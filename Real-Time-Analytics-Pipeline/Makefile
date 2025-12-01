.PHONY: help start stop restart logs clean test

help:
	@echo "Real-Time Analytics Pipeline - Make Commands"
	@echo ""
	@echo "  make start       - Start all services"
	@echo "  make stop        - Stop all services"
	@echo "  make restart     - Restart all services"
	@echo "  make logs        - View logs"
	@echo "  make clean       - Clean all data and containers"
	@echo "  make test        - Run tests"
	@echo "  make generate    - Generate sample events"

start:
	docker-compose up -d
	@echo "✅ Services started. Dashboard: http://localhost:3000"

stop:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -f

test:
	pytest tests/

generate:
	python scripts/generate_events.py --rate 1000 --duration 60
