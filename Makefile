.PHONY: test nd

foo:
	echo "boooo"

test-rust:
	cd nd && cargo test

nd:
	cd nd && maturin develop

test: test-rust nd
	pytest tests -v
