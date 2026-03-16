.PHONY: install test lint docker
install:
	pip install -e backend[dev]
test:
	pytest backend/tests/ -q
lint:
	ruff check backend/
docker:
	docker-compose up --build
