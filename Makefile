.PHONY: test

foo:
	@echo "boooo, don't just type `make` into an unknown repository booo"

test-rust:
	cargo test
	cargo test --benches

nd-dev:
	pip install -e.

install:
	pip install .

test-py: nd-dev
	pytest tests -v

test: test-rust test-py
