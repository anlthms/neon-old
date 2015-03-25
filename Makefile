# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
# Top-level control of the building/installation/cleaning of various targets

# these variables control the type of build, use -e to override their default
# values, which are defined in setup.cfg
DEV := $(strip $(shell grep -i '^ *DEV *=' setup.cfg | cut -f 2 -d '='))
CPU := $(strip $(shell grep -i '^ *CPU *=' setup.cfg | cut -f 2 -d '='))
GPU := $(strip $(shell grep -i '^ *GPU *=' setup.cfg | cut -f 2 -d '='))
DIST := $(strip $(shell grep -i '^ *DIST *=' setup.cfg | cut -f 2 -d '='))

# get release version info
RELEASE := $(strip $(shell grep '^VERSION *=' setup.py | cut -f 2 -d '=' \
	                         | tr -d "\'"))

# these variables control where we publish Sphinx docs to
DOC_DIR := doc
DOC_PUB_HOST := stagecoach.dreamhost.com
DOC_PUB_USER := amikho1
DOC_PUB_PATH := /home/amikho1/framework.nervanasys.com/docs
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)

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
INSTALL_REQUIRES :=
ifeq ($(DEV), 0)
  NOSE_ATTRS := $(NOSE_ATTRS),'!dev'
else
  INSTALL_REQUIRES := $(INSTALL_REQUIRES) 'nose>=1.3.0' 'cython>=0.19.1' \
		'flake8>=2.2.2' 'pep8-naming>=0.2.2' 'sphinx>=1.2.2' \
		'sphinxcontrib-napoleon>=0.2.8' 'scikit-learn>=0.15.2' 'matplotlib>=1.4.0'
  INSTALL_REQUIRES := $(INSTALL_REQUIRES) \
    'git+http://gitlab.localdomain/algorithms/imgworker.git\#egg=imgworker>=0.2.1'
endif
ifeq ($(GPU), 0)
  NOSE_ATTRS := $(NOSE_ATTRS),'!cuda'
else
  INSTALL_REQUIRES := $(INSTALL_REQUIRES) \
    'git+https://github.com/NervanaSystems/cuda-convnet2.git\#egg=cudanet>=0.2.5'
endif
ifeq ($(DIST), 0)
  NOSE_ATTRS := $(NOSE_ATTRS),'!dist'
else
  INSTALL_REQUIRES := $(INSTALL_REQUIRES) 'mpi4py>=1.3.1'
endif

.PHONY: default build develop install uninstall test test_all sanity speed \
	      grad all clean_pyc clean doc html style lint bench dist publish_doc \
	      insert_compiler_hints strip_compiler_hints release

default: build

build: clean_pyc
	@echo "Running build(DEV=$(DEV) CPU=$(CPU) GPU=$(GPU) DIST=$(DIST))..."
	@python setup.py neon --dev $(DEV) --cpu $(CPU) --gpu $(GPU) --dist $(DIST) \
		build_ext --inplace

develop: clean_pyc .git/hooks/pre-commit
	@echo "Running develop(DEV=$(DEV) CPU=$(CPU) GPU=$(GPU) DIST=$(DIST))..."
	@python setup.py neon --dev $(DEV) --cpu $(CPU) --gpu $(GPU) --dist $(DIST) \
		develop

# unfortunately there is no way to communicate custom commands into pip
# install, hence having to specify installation requirements twice (once
# above, and once inside setup.py). Ugly kludge, but seems like the only way
# to support python setup.py install and pip install.
# Since numpy is required for building some of the other dependent packages
# we need to separately install it first
install: clean_pyc
	@echo "Running install..."
	@pip install 'numpy>=1.8.1' 'PyYAML>=3.11'
	@pip install $(INSTALL_REQUIRES) .

uninstall:
	@echo "Running uninstall..."
	@pip uninstall -y neon

test: build
	@echo "Running unit tests..."
	nosetests $(NOSE_ATTRS) $(NOSE_FLAGS) neon

test_all:
	@echo "Running test_all..."
	@tox -- -e CPU=$(CPU) GPU=$(GPU) DIST=$(DIST)

sanity: build
	@echo "Running sanity checks..."
	@PYTHONPATH=${PYTHONPATH}:./ python neon/tests/sanity_check.py \
		--cpu $(CPU) --gpu $(GPU) --datapar $(DIST) --modelpar $(DIST)

speed: build
	@echo "Running speed checks..."
	@PYTHONPATH=${PYTHONPATH}:./ python neon/tests/speed_check.py \
		--cpu $(CPU) --gpu $(GPU) --datapar $(DIST) --modelpar $(DIST)

grad: build
	@echo "Running gradient checks..."
ifeq ($(CPU), 1)
	@echo "CPU:"
	@PYTHONPATH=${PYTHONPATH}:./ bin/grad \
		examples/convnet/synthetic-sanity_check.yaml
endif
ifeq ($(GPU), 1)
	@echo "GPU:"
	@PYTHONPATH=${PYTHONPATH}:./ bin/grad --gpu \
		examples/convnet/synthetic-sanity_check.yaml
endif

all: style test sanity grad speed

clean_pyc:
	@-find . -name '*.py[co]' -exec rm {} \;

clean:
	-python setup.py clean
	-rm -f neon/backends/flexpt_dtype.so
	-rm -f neon/backends/flexpt_cython.so

insert_compiler_hints:
	@echo "Preprocessing source code to insert compiler hints..."
	-python neon/util/compiler_hints.py

strip_compiler_hints:
	@echo "Preprocessing source code to remove compiler hints..."
	-python neon/util/compiler_hints.py -s

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
		rsync -avz -essh --perms --chmod=ugo+rX . \
		$(DOC_PUB_USER)@$(DOC_PUB_HOST):$(DOC_PUB_RELEASE_PATH)
	@-ssh $(DOC_PUB_USER)@$(DOC_PUB_HOST) \
		'rm -f $(DOC_PUB_PATH)/latest && \
		 ln -sf $(DOC_PUB_RELEASE_PATH) $(DOC_PUB_PATH)/latest'

release: publish_doc
	@gitchangelog > ChangeLog
