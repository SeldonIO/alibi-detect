.PHONY: install-dev
install-dev:
	pip install -r requirements/dev.txt

.PHONY: install
install:
	pip install -e .[all]

# Note: The pytest-randomly seed is fixed at 0 for now. Once the legacy np.random.seed(0)'s
# are removed from tests, this can be removed, allowing all tests to use random seeds.
.PHONY: test
test:
	TF_USE_LEGACY_KERAS=1 pytest --randomly-seed=0 alibi_detect/utils/tests/test_saving_legacy.py
	pytest --randomly-seed=0 --ignore=alibi_detect/utils/tests/test_saving_legacy.py alibi_detect

.PHONY: lint
lint: ## Check linting according to the flake8 configuration in setup.cfg
	flake8 .

.PHONY: mypy
mypy: ## Run typeckecking according to mypy configuration in setup.cfg
	mypy .

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

.PHONY: build_pypi
build_pypi: ## Build the Python package
	python setup.py sdist bdist_wheel

.PHONY: push_pypi_test
push_pypi_test: ## Upload the Python package to the test PyPI registry
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: push_pypi
push_pypi: ## Upload the Python package to the PyPI registry
	twine upload dist/*

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

