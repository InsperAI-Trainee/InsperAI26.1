MKDOCS_DEV_ADDR ?= 127.0.0.1:8000

dev:
	cd handout && uv run python dev_server.py --dev-addr $(MKDOCS_DEV_ADDR)

build:
	cd handout && uv run mkdocs build --strict

deploy:
	cd handout && uv run mkdocs gh-deploy --force
