# Development tasks. Python runs through uv and JavaScript through npm, so no environment needs
# to be activated first. Python tools start with python -m instead of their .exe launchers,
# which Windows application control policies can block.

UV ?= uv
NPM ?= npm
PORT ?= 8000

.PHONY: help install data train docs lint format test e2e play screenshots

help:
	@echo install      Install Python and JavaScript dependencies and the pre-commit hooks
	@echo data         Download the HaGRID annotations and the recorded human games
	@echo train        Train the gesture reader, tune the predictor, and write the game files
	@echo docs         The same, then copy the report and figures into docs/
	@echo lint         Run ruff, mypy, Biome, and the JavaScript type check
	@echo format       Apply ruff and Biome fixes and formatting
	@echo test         Run the Python and JavaScript tests
	@echo e2e          Play the game in a headless browser
	@echo play         Serve the game on http://127.0.0.1:8000
	@echo screenshots  Capture the README screenshots with a headless browser

install:
	$(UV) sync --frozen
	$(NPM) ci
	$(UV) run python -m pre_commit install

data:
	$(UV) run python scripts/download_data.py

train:
	$(UV) run python scripts/train_and_evaluate.py

docs:
	$(UV) run python scripts/train_and_evaluate.py --publish_docs

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run python -m mypy
	$(NPM) run check

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .
	$(NPM) run format

test:
	$(UV) run python -m pytest --cov --cov-report=term-missing
	$(NPM) test

e2e:
	$(NPM) run e2e

play:
	$(UV) run python -m http.server $(PORT) --bind 127.0.0.1 --directory web

screenshots:
	node web/e2e/screenshots.js docs/figures
