
VERSION ?= $(shell git describe --tags --always)

define INIT_FILE

__version__ = "$(VERSION)"
endef
export INIT_FILE

all: build

version:
	echo $$INIT_FILE > fylearn/__init__.py

install_flit:
	pip install flit

build: install_flit version
	flit build

clean:
	rm -rf dist .pytest_cache build .eggs fylearn.egg-info htmlcov .tox
