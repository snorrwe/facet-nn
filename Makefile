.PHONY: test

foo:
	echo "boooo"

test-rust:
	cd nd && cargo test

build-nd:
	cd nd && maturin develop

test: test-rust build-nd
	pytest tests -v
