.PHONY: test nd

foo:
	echo "boooo"

test-rust:
	cargo test

nd-dev:
	cd pyfacet && maturin develop

nd:
	cd pyfacet && maturin build --release
	pip install target/wheels/pyfacet-0.1.0-cp39-none-win_amd64.whl --upgrade


test-py: nd-dev
	pytest tests -v

test: test-rust test-py
