.PHONY: help init build up down logs ps smoke clean-outputs

help:
	@echo "init          - create local .env and /data dirs"
	@echo "build         - docker compose build"
	@echo "up            - start the stack detached"
	@echo "down          - stop the stack"
	@echo "logs          - follow all logs"
	@echo "ps            - service status"
	@echo "smoke         - run scripts/smoke_test.py against the gateway"

init:
	@test -f .env || cp .env.example .env
	@. ./.env 2>/dev/null; mkdir -p "$${DATA_ROOT:-/data/expertasd_tts_pipeline}/outputs" \
		"$${DATA_ROOT:-/data/expertasd_tts_pipeline}/redis-data"
	@echo "Initialized. Review .env before 'make build'."

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

smoke:
	python3 scripts/smoke_test.py

clean-outputs:
	@echo "Manual only. See docs/runbook.md before deleting from the shared volume."
