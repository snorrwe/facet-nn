.PHONY: test nd

foo:
	echo "boooo"

check-rust:
	cargo check
	cargo clippy

test-rust:
	cargo test

nd-dev:
	cd pydu && maturin develop

nd:
	cd pydu && maturin build --release
	pip install target/wheels/pydu-0.1.0-cp39-none-win_amd64.whl --upgrade


test-py: nd-dev
	pytest tests -v

test: check-rust test-rust test-py
