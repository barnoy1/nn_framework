.PHONY: changelog changelog-incremental release

changelog:
	uv run cz changelog

changelog-incremental:
	uv run cz changelog --incremental

release:
	uv run cz bump --changelog
