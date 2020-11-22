.PHONY: test nd

foo:
	echo "boooo"

check-rust:
	cd nd && cargo check

test-rust:
	cd nd && cargo test

nd:
	cd nd && maturin develop

test: check-rust test-rust nd
	pytest tests -v
