# Top-level control of the building/installation/cleaning of various targets

DOC_DIR=doc

.PHONY: default build develop clean_pyc clean doc html

default: build

build: clean_pyc
	python setup.py build_ext --inplace

develop: build
	-python setup.py develop

install: build
	python setup.py install

clean_pyc:
	-find . -name '*.py[co]' -exec rm {} \;

clean:
	-python setup.py clean

doc: build
	$(MAKE) -C $(DOC_DIR) clean
	$(MAKE) -C $(DOC_DIR) html

html: doc

publish_doc: doc
	-cd $(DOC_DIR)/build/html && python -m SimpleHTTPServer
