
.PHONY: clean build install editable-install test


clean:
	rm -rf build
	rm -rf dist
	rm -rf qmodel.egg-info
	rm -f qmodel/data/cpp/*cpython*.so

build: clean
	python -m build --wheel

install: clean
	make build
	pip install dist/qmodel-*.whl

editable-install: clean
	pip install -e .

test:
	pytest -q
