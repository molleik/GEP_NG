RUN_COMMAND = poetry run


### Formatting ###

# Format python files with the industry-standard Black code formatter.
# See https://black.readthedocs.io/en/stable/
format:
	$(RUN_COMMAND) black --include ".py$$" --extend-exclude ".venv/" .


### Testing ###

lint: 
	$(RUN_COMMAND) black --check --include ".py$$" --extend-exclude ".venv/" .
	$(RUN_COMMAND) mypy .

### Dependency Management ###

# Install all the project dependencies in a virtual environment managed by poetry.
# See https://python-poetry.org/docs/

dependencies:
	pip show -q poetry || pip install poetry
	poetry install






