SHELL = /bin/bash

pkg_name = aitelier
src_pkg = src/$(pkg_name)
.PHONY: format install

format:
	ruff . --fix
	mypy .
	black .

install:
	pip install -e .

clean:
	-rm -rf .mypy_cache
	-rm -rf build
	-rm -rf .eggs
	-rm -rf dist
	-rm -rf *.egg-info
	-rm -rf .ruff_cache