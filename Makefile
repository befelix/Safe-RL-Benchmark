module="SafeRLBench"

GREEN=\033[0;32m
NC=\033[0m

# Flake 8 ignore errors
flakeignore='E402,W503'

# Pydocstyle ignore errors
pydocignore='D105'

style:
	@echo "${GREEN}Running style tests:${NC}"
	@flake8 ${module} --exclude test*.py,__init__.py --show-source
	@flake8 ${module} --filename=__init__.py,test*.py --ignore=F --show-source

docstyle:
	@echo "${GREEN}Testing docstring conventions:${NC}"
	@pydocstyle ${module} --match='(?!__init__).*\.py' 2>&1 | grep -v "WARNING: __all__"

unittests2:
	@echo "${GREEN}Running unit tests for 2.7:${NC}"
	@nosetests-2.7 -v --with-doctest --with-coverage --cover-erase --cover-package=${module} ${module}

unittests3:
	@echo "${GREEN}Running unit tests for 3.5:${NC}"
	@nosetests-3.5 -v --with-doctest --with-coverage --cover-erase --cover-package=${module} ${module}

unittests: unittests2 unittests3

test: style docstyle unittests

history:
	git log --graph --decorate --oneline

clean:
	find . -type f -name '*.pyc' -exec rm -f {} ';'
