.PHONY: test

foo:
	echo "boooo"


test:
	cd nd && cargo test && maturin develop
	pytest tests -v
