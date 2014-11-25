# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
# Top-level control of the building/installation/cleaning of various targets

# these variables control the type of build, use -e to override their default
# values, which are defined in setup.cfg
DEV := $(shell grep '^ *DEV *=' setup.cfg | cut -f 2 -d '=')
CPU := $(shell grep '^ *CPU *=' setup.cfg | cut -f 2 -d '=')
GPU := $(shell grep '^ *GPU *=' setup.cfg | cut -f 2 -d '=')
DIST := $(shell grep '^ *DIST *=' setup.cfg | cut -f 2 -d '=')

# these variables control where we publish Sphinx docs to
DOC_DIR := doc
DOC_PUB_HOST := atlas.localdomain
DOC_PUB_USER := neon
DOC_PUB_PATH := /home/neon/public/

# these control test options and attribute filters
NOSE_FLAGS := ""  # --pdb --pdb-failures
NOSE_ATTRS := -a '!slow'

# ensure a cuda capable GPU is installed
ifeq ($(GPU), 1)
  ifeq ($(shell uname -s), Darwin)
    ifneq ($(shell kextstat | grep -i cuda > /dev/null 2>&1; echo $$?), 0)
      $(info No CUDA capable GPU installed on OSX.  Forcing GPU=0)
      override GPU := 0
    endif
  else
    # we assume a Linux-like OS
    ifneq ($(shell nvidia-smi > /dev/null 2>&1; echo $$?), 0)
      $(info No CUDA capable GPU installed.  Forcing GPU=0)
      override GPU := 0
    endif
  endif
endif

# update options based on build type
PYSETUP_FLAGS := ""
ifeq ($(GPU), 0)
  NOSE_ATTRS := $(NOSE_ATTRS),'!cuda'
else
  PYSETUP_FLAGS := $(PYSETUP_FLAGS) --gpu
endif
ifeq ($(DIST), 0)
  NOSE_ATTRS := $(NOSE_ATTRS),'!dist'
else
  PYSETUP_FLAGS := $(PYSETUP_FLAGS) --dist
endif

.PHONY: default build develop install uninstall test test_all sanity speed \
	      grad all clean_pyc clean doc html style lint bench dist publish_doc \
	      release

default: build

build: clean_pyc
	@echo "Running build(DEV=$(DEV) CPU=$(CPU) GPU=$(GPU) DIST=$(DIST))..."
	@python setup.py neon --dev $(DEV) --cpu $(CPU) --gpu $(GPU) --dist $(DIST) \
		build_ext --inplace

develop: build .git/hooks/pre-commit
	@echo "Running develop..."
	@python setup.py neon --dev $(DEV) --cpu $(CPU) --gpu $(GPU) --dist $(DIST) \
		develop

install: build
	@echo "Running install..."
	@pip install .

uninstall:
	@echo "Running uninstall..."
	@pip uninstall -y neon

test: build
	@echo "Running unit tests..."
	nosetests $(NOSE_ATTRS) $(NOSE_FLAGS) neon

test_all: build
	@echo "Running test_all..."
	@tox -- -e CPU=$(CPU) GPU=$(GPU) DIST=$(DIST)

sanity: build
	@echo "Running sanity checks..."
	@PYTHONPATH=${PYTHONPATH}:./ python neon/tests/sanity_check.py \
		--cpu $(CPU) --gpu $(GPU) --dist $(DIST)

speed: build
	@echo "Running speed checks..."
	@PYTHONPATH=${PYTHONPATH}:./ python neon/tests/speed_check.py \
		--cpu $(CPU) --gpu $(GPU) --dist $(DIST)

grad: build
	@echo "Running gradient checks..."
ifeq ($(CPU), 1)
	@echo "CPU:"
	@PYTHONPATH=${PYTHONPATH}:./ bin/grad neon/tests/check_cpu.yaml
endif
ifeq ($(GPU), 1)
	@echo "GPU:"
	@PYTHONPATH=${PYTHONPATH}:./ bin/grad neon/tests/check_gpu.yaml
endif

all: style test sanity grad speed

clean_pyc:
	@-find . -name '*.py[co]' -exec rm {} \;

clean:
	-python setup.py clean
	-rm -f neon/backends/flexpt_dtype.so
	-rm -f neon/backends/flexpt_cython.so

doc: build
	$(MAKE) -C $(DOC_DIR) clean
	$(MAKE) -C $(DOC_DIR) html

html: doc

style:
	@-flake8 --exclude=.tox,build,dist,src .

.git/hooks/pre-commit:
	@flake8 --install-hook
	@-touch .git/hooks/pre-commit

lint:
	@-pylint --output-format=colorized neon

bench: build
	@PYTHONPATH="." benchmarks/run_benchmarks.py

dist:
	@python setup.py sdist

publish_doc: doc
	@-cd $(DOC_DIR)/build/html && \
		rsync -avz -essh --perms --chmod=ugo+rX . $(DOC_PUB_USER)@$(DOC_PUB_HOST):$(DOC_PUB_PATH)

release: publish_doc
	@gitchangelog > ChangeLog
