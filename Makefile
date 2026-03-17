.PHONY: formatting changelog cz_commit release

formatting:
	python3 -m pre_commit run  --all-files
	
changelog:
	uv run cz changelog
	uv run cz changelog --incremental
cz_commit:
	uv run cz commit
	
release:
	uv run cz bump --changelog

bump_patch:
	uv run cz bump --increment PATCH

bump_minor:
	uv run cz bump --increment MINOR

bump_major:
	uv run cz bump --increment MAJOR