.PHONY: test nd

foo:
	echo "boooo"

check-rust:
	cargo check

test-rust:
	cargo test

nd:
	cd pydu && maturin develop

test: check-rust test-rust nd
	pytest tests -v
