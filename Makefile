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




