![Build and Test](https://github.com/snorrwe/facet-nn/workflows/Build%20and%20Test/badge.svg)

## Install deps

```sh
python -m venv env
source env/scripts/activate

pip install -r requirements.txt
```

## Build & Install the library in the current Python (virtual) environment

```sh
make install
```


## Running the test suite

```sh
make test
```


## Project layout

```txt
|- facet-core  # Core, Rust-only linalg / nn code
|- pyfacet     # The Python library using facet-core
  |- pyfacet   # Python code
  |- src/      # Rust code interfacing between Python and facet-core
|- tests       # Python tests
|- models      # Actual models built on top of this lib
```
