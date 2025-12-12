
VERSION_FILE = fylearn/_version.py
FLIT_INDEX_URL ?= https://test.pypi.org/legacy/
FLIT_USERNAME = __token__
FLIT_PASSWORD ?= dummy

.PHONY: version

all: build

deps:
	uv sync

version: deps
	uv run python .ci/versioning.py

wheel: version deps
	uv build

build: wheel

publish:
	@FLIT_USERNAME=$(FLIT_USERNAME) FLIT_PASSWORD=$(FLIT_PASSWORD) FLIT_INDEX_URL=$(FLIT_INDEX_URL) flit publish --format wheel
	@FLIT_USERNAME=$(FLIT_USERNAME) FLIT_PASSWORD=$(FLIT_PASSWORD) FLIT_INDEX_URL=$(FLIT_INDEX_URL) flit publish --format sdist

test:
	uv tool install tox --with tox-uv
	tox

clean:
	rm -rf dist .pytest_cache build .eggs fylearn.egg-info htmlcov .tox *.whl *~ fylearn/*~ fylearn/_version.py
