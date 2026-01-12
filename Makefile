
VERSION_FILE = fylearn/_version.py
FLIT_INDEX_URL ?= $(or $(shell grep '^FLIT_INDEX_URL=' .env 2>/dev/null | tail -n 1 | cut -d= -f2-),https://test.pypi.org/legacy/)
FLIT_USERNAME ?= $(or $(shell grep '^FLIT_USERNAME=' .env 2>/dev/null | tail -n 1 | cut -d= -f2-),__token__)
FLIT_PASSWORD ?= $(or $(shell grep '^FLIT_PASSWORD=' .env 2>/dev/null | tail -n 1 | cut -d= -f2-),dummy)

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
	uv publish --token $(FLIT_PASSWORD) --publish-url $(FLIT_INDEX_URL)

test:
	uv tool install tox --with tox-uv
	tox

clean:
	rm -rf dist .pytest_cache build .eggs fylearn.egg-info htmlcov .tox *.whl *~ fylearn/*~ fylearn/_version.py
