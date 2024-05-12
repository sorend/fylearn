

all: build

install_flit:
	pip install flit

build: install_flit
	flit build

clean:
	rm -rf dist .pytest_cache build .eggs fylearn.egg-info htmlcov .tox
