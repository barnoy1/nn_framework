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
