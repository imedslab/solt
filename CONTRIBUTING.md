# Contribution Guidelines

## Setting up the environment
The project is managed with [uv](https://docs.astral.sh/uv/). Install it first
(e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`), then run:

```
uv sync
uv run pre-commit install
```

`uv sync` creates a virtual environment in `.venv`, installs `solt` in editable mode together with
the `dev` dependency group (tests and linters), and resolves exactly the versions pinned in `uv.lock`.
Prefix commands with `uv run` (e.g. `uv run pytest tests`) or activate the environment with
`source .venv/bin/activate`.

To use a specific Python version, pass it explicitly, e.g. `uv sync --python 3.11`.

If you change dependencies in `pyproject.toml`, run `uv lock` and commit the updated `uv.lock`.

### Building the documentation
The docs need [pandoc](https://pandoc.org/installing.html) installed on your system
(`brew install pandoc` or `sudo apt-get install pandoc`). Then:

```
uv sync --group docs
cd doc && uv run make html
```

## Contributing

1. Implement the augmentation
2. Write tests
3. Make sure that your codecoverage is 100%. The criteria for merging into master is
that all of the tests pass and that the codecoverage is 100%.

To run the tests with coverage:
```
uv run pytest tests --cov solt
```

Before commiting commiting, execute these actions:
```
git add .
uv run black --config=black.toml .
uv run pre-commit
```

If everything is alright, this will be the expected output:
```
black....................................................................Passed
flake8...................................................................Passed
check for case conflicts.................................................Passed
check for broken symlinks............................(no files to check)Skipped
fix end of files.........................................................Passed
fix requirements.txt.................................(no files to check)Skipped
```
