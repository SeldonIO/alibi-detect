IMAGE=alibi-detect-server
TAG=0.0.1

.PHONY: install_dev
install_dev:
	pip install -e . -r requirements-dev.txt

.PHONY: type_check
type_check:
	mypy --ignore-missing-imports adserver

.PHONY: test
test: type_check
	pytest -W ignore

.PHONY: lint
lint:
	black .

docker-build: 
	docker build -f Dockerfile -t seldonio/${IMAGE}:${TAG} .

docker-push:
	docker push seldonio/${IMAGE}:${TAG} 
