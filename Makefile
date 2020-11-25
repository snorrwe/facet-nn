.PHONY: test nd

foo:
	echo "boooo"

check-rust:
	cargo check

test-rust:
	cargo test

nd-dev:
	cd pydu && maturin develop

nd:
	cd pydu && maturin develop --release

test: check-rust test-rust nd-dev
	pytest tests -v
