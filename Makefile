.PHONY: test

foo:
	@echo "boooo, don't just type `make` into an unknown repository booo"

test-rust:
	cargo test
	cargo test --benches

nd-dev:
	cd pyfacet && maturin develop

nd-release:
	cd pyfacet && maturin develop --release

install:
	cd pyfacet && maturin build --release
	pip install target/wheels/pyfacet-0.1.0-cp39-none-win_amd64.whl --upgrade


test-py: nd-dev
	pytest tests -v

test: test-rust test-py
