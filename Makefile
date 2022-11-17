.PHONY: install
install: ## Install package in editable mode with all the dependencies
	pip install -e .

.PHONY: clean
clean:	# Clean up install
	pip uninstall -y alibi-detect
	rm -r alibi_detect.egg-info/ dist/ build/

.PHONY: test
test: ## Run all tests
	pytest alibi_detect

.PHONY: lint
lint: ## Check linting according to the flake8 configuration in setup.cfg
	flake8 .

.PHONY: mypy
mypy: ## Run typeckecking according to mypy configuration in setup.cfg
	mypy .

.PHONY: build
build: ## Build the Python package (for debugging, building for release is done via a GitHub Action)
	# Below commands are a duplicate of those in the publish.yml "Build" step
	# (virtualenv added due to Debian issue https://github.com/pypa/build/issues/224)
	python -m pip install --upgrade pip virtualenv
	python -m pip install build~=0.9
	python -m build --sdist --wheel --outdir dist/ .

.PHONY: build_docs
build_docs:
	# readthedocs.org build command
	python -m sphinx -T -b html -d doc/_build/doctrees -D language=en doc/source  doc/_build/html

.PHONY: build_latex
build_latex: ## Build the documentation into a pdf
	# readthedocs.org build command
	# explicit cd here due to a bug in latexmk 4.41
	python -m sphinx -b latex -d doc/_build/doctrees -D language=en doc/source doc/_build/latex && \
	cd doc/_build/latex && \
	latexmk -pdf -f -dvi- -ps- -jobname=alibi-detect -interaction=nonstopmode

.PHONY: clean_docs
clean_docs: ## Clean the documentation build
	$(MAKE) -C doc clean
	rm -r doc/source/api

.PHONY: help
help: ## Print out help message on using these commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: licenses
licenses:
	# create a tox environment and pull in license information
	tox --recreate -e licenses
	cut -d, -f1,3 ./licenses/license_info.csv \
					> ./licenses/license_info.no_versions.csv

.PHONY: check_licenses
	# check if there has been a change in license information, used in CI
check_licenses:
	git --no-pager diff --exit-code ./licenses/license_info.no_versions.csv

.PHONY: repl
tox-env=default
repl:
	env COMMAND="python" tox -e $(tox-env)
