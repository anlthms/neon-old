# Top-level control of the building/installation/cleaning of various targets

DOC_DIR=doc
DOC_PUB_HOST=atlas.localdomain
DOC_PUB_USER=neon
DOC_PUB_PATH=/home/neon/public/

NOSE_FLAGS=""  # --pdb --pdb-failures

# check if a cuda capable GPU is installed
NO_CUDA_GPU=1
# NO_CUDA_GPU set to 0 will enable GPU based backend tests, which we attempt to
# infer automatically
ifeq ($(shell uname -s),Darwin)
	# OSX checking for CUDA drivers
	NO_CUDA_GPU=$(shell kextstat | grep -i cuda > /dev/null 2>&1; echo $$?)
else
	# Assume something Linux'y
	NO_CUDA_GPU=$(shell nvidia-smi > /dev/null 2>&1; echo $$?)
endif


.PHONY: default build develop install uninstall test test_all clean_pyc clean \
	      doc html style lint bench dist publish_doc release

default: build

build: clean_pyc
	python setup.py build_ext --inplace

develop: build .git/hooks/pre-commit
	-python setup.py develop

install: build
	pip install --upgrade --download-cache .pkgs --requirement requirements.txt .

uninstall:
	pip uninstall -y neon

test: build
ifeq ($(NO_CUDA_GPU),0)
	nosetests -a '!slow' $(NOSE_FLAGS) neon
else
	@echo "No CUDA compatible GPU found, disabling GPU tests"
	nosetests -a '!slow','!cuda' $(NOSE_FLAGS) neon
endif

test_all: build
	tox

clean_pyc:
	-find . -name '*.py[co]' -exec rm {} \;

clean:
	-python setup.py clean
	-rm -f neon/backends/fixpt_dtype.so
	-rm -f neon/backends/fixpt_cython.so

doc: build
	$(MAKE) -C $(DOC_DIR) clean
	$(MAKE) -C $(DOC_DIR) html

html: doc

style:
	-flake8 --exclude=.tox,build,dist,src .

.git/hooks/pre-commit:
	-flake8 --install-hook
	-touch .git/hooks/pre-commit

lint:
	-pylint --output-format=colorized neon

bench: build
	PYTHONPATH="." benchmarks/run_benchmarks.py

dist:
	python setup.py sdist

publish_doc: doc
	-cd $(DOC_DIR)/build/html && \
		rsync -avz -essh . $(DOC_PUB_USER)@$(DOC_PUB_HOST):$(DOC_PUB_PATH)

release: publish_doc
	gitchangelog > ChangeLog
