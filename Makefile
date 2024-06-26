
VERSION_FILE = .ci/_version.py
FLIT_INDEX_URL ?= https://test.pypi.org/legacy/
FLIT_USERNAME = __token__
FLIT_PASSWORD ?= dummy

all: build

deps:
	pip install versiontag flit

$(VERSION_FILE): deps
	python .ci/versioning.py

local_install: $(VERSION_FILE) deps
	pip install -e '.[dev]'

wheel: local_install
	flit build --format wheel
	flit build --format sdist

build: wheel

publish:
	@FLIT_USERNAME=$(FLIT_USERNAME) FLIT_PASSWORD=$(FLIT_PASSWORD) FLIT_INDEX_URL=$(FLIT_INDEX_URL) flit publish --format wheel
	@FLIT_USERNAME=$(FLIT_USERNAME) FLIT_PASSWORD=$(FLIT_PASSWORD) FLIT_INDEX_URL=$(FLIT_INDEX_URL) flit publish --format sdist

test: local_install
	tox

clean:
	rm -rf dist .pytest_cache build .eggs fylearn.egg-info htmlcov .tox *.whl *~ fylearn/*~ fylearn/_version.py
