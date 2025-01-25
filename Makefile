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