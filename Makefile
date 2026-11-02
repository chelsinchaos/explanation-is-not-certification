.PHONY: test figures paper clean

test:
	python3 -m pytest -q tests/

figures:
	python3 scripts/make_figures.py

paper: figures
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	cd paper && latexmk -C && rm -f main.txt build.log
	rm -rf .pytest_cache src/selcert/__pycache__ tests/__pycache__ scripts/__pycache__
