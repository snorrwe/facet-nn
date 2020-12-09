## Install deps

```sh
python -m venv env
source env/scripts/activate

pip install -r requirements.txt
```

## Build & Install the library

```sh
make nd
```


## Running the test suite

```sh
make test
```


## Code layout

```txt
|- facet-core  # Core, Rust-only linalg / nn code
|- pyfacet     # The Python library using facet-core
  |- pyfacet   # Python code
  |- src/      # Rust code interfacing between Python and facet-core
|- tests       # Python tests
|- models      # Actual models built on top of this lib
```
