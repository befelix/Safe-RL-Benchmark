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
	@nosetests-2.7 -v --with-doctest --cover-erase --cover-package=${module} ${module}

unittests3:
	@echo "${GREEN}Running unit tests for 3.5:${NC}"
	@nosetests-3.5 -v --with-doctest --with-coverage --cover-erase --cover-package=${module} ${module}

coverage: unittests3
	@echo "${GREEN}Create coverage report:${NC}"
	@coverage html

unittests: unittests2 unittests3

test: style docstyle unittests

setup_docker2:
	docker build -f misc/Dockerfile.python2 -t srlb-py27-image .

setup_docker3:
	docker build -f misc/Dockerfile.python3 -t srlb-py35-image .

docker2:
	@echo "${GREEN}Running unit tests for 2.7:${NC}"
	docker run -v $(shell pwd):/code/ srlb-py27-image nosetests --with-doctest --verbosity=2 SafeRLBench

docker3:
	@echo "${GREEN}Running unit tests for 3.5:${NC}"
	docker run -v $(shell pwd):/code/ srlb-py35-image nosetests --with-doctest --verbosity=2 SafeRLBench

history:
	git log --graph --decorate --oneline

clean:
	find . -type f -name '*.pyc' -exec rm -f {} ';'
	rm -r htmlcov
