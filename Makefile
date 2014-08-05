# Top-level control of the building/installation/cleaning of various targets

DOC_DIR=doc
DOC_PUB_HOST=192.168.20.2
DOC_PUB_USER=mylearn
DOC_PUB_PATH=/home/mylearn/public/

.PHONY: default build develop clean_pyc clean doc html test dist publish_doc \
	      benchmark test_all style lint

default: build

build: clean_pyc
	python setup.py build_ext --inplace

develop: build
	-python setup.py develop

install: build
	pip install .

uninstall:
	pip uninstall -y mylearn

test: build
	nosetests -a '!slow','!cuda' mylearn

test_all: build
	tox

clean_pyc:
	-find . -name '*.py[co]' -exec rm {} \;

clean:
	-python setup.py clean

doc: build
	$(MAKE) -C $(DOC_DIR) clean
	$(MAKE) -C $(DOC_DIR) html

html: doc

style:
	-flake8 .

lint:
	-pylint --output-format=colorized mylearn

bench: build
	PYTHONPATH="." benchmarks/run_benchmarks.py

dist:
	python setup.py sdist

publish_doc: doc
	-cd $(DOC_DIR)/build/html && \
		rsync -avz -essh . $(DOC_PUB_USER)@$(DOC_PUB_HOST):$(DOC_PUB_PATH)
