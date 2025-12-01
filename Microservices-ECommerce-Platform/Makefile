.PHONY: help build deploy test clean

help:
	@echo "Microservices E-Commerce Platform - Make Commands"
	@echo ""
	@echo "  make build           - Build all Docker images"
	@echo "  make deploy-local    - Deploy to local Kubernetes"
	@echo "  make deploy-dev      - Deploy to development"
	@echo "  make deploy-prod     - Deploy to production"
	@echo "  make test-unit       - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make clean           - Clean up resources"
	@echo "  make logs            - View logs"

build:
	docker build -t product-service:latest ./services/product-service
	docker build -t order-service:latest ./services/order-service
	docker build -t payment-service:latest ./services/payment-service
	docker build -t notification-service:latest ./services/notification-service

deploy-local:
	kubectl create namespace ecommerce || true
	helm install -n ecommerce infrastructure ./helm/infrastructure
	helm install -n ecommerce ecommerce ./helm/ecommerce

deploy-dev:
	helm upgrade --install -n ecommerce-dev infrastructure ./helm/infrastructure -f ./helm/values-dev.yaml
	helm upgrade --install -n ecommerce-dev ecommerce ./helm/ecommerce -f ./helm/values-dev.yaml

deploy-prod:
	helm upgrade --install -n ecommerce-prod infrastructure ./helm/infrastructure -f ./helm/values-prod.yaml
	helm upgrade --install -n ecommerce-prod ecommerce ./helm/ecommerce -f ./helm/values-prod.yaml

test-unit:
	cd services/order-service && pytest tests/unit -v
	cd services/notification-service && pytest tests/unit -v

test-integration:
	cd services/order-service && pytest tests/integration -v

clean:
	helm uninstall -n ecommerce infrastructure || true
	helm uninstall -n ecommerce ecommerce || true
	kubectl delete namespace ecommerce || true

logs:
	kubectl logs -n ecommerce -l app=order-service --tail=100 -f
