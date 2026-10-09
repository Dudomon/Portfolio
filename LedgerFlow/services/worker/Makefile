BINARY := bin/worker
PKG    := ./...

.PHONY: all tidy build test cover lint run docker clean

all: lint test build

tidy:
	go mod tidy

build:
	CGO_ENABLED=0 go build -trimpath -o $(BINARY) ./cmd/worker

test:
	go test -race -count=1 $(PKG)

cover:
	go test -count=1 -coverprofile=coverage.out $(PKG)
	go tool cover -func=coverage.out

lint:
	golangci-lint run

run:
	set -a && . ./.env && set +a && go run ./cmd/worker

docker:
	docker build -t ledgerflow-worker:dev .

clean:
	rm -rf bin coverage.out
