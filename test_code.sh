#!/usr/bin/env bash

module="SafeRLBench"

get_script_dir () {
     SOURCE="${BASH_SOURCE[0]}"
     # While $SOURCE is a symlink, resolve it
     while [ -h "$SOURCE" ]; do
          DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
          SOURCE="$( readlink "$SOURCE" )"
          # If $SOURCE was a relative symlink (so no "/" as prefix, need to resolve it relative to the symlink base directory
          [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
     done
     DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
     echo "$DIR"
}

# tensorflow environment variable
export TF_CPP_MIN_LOG_LEVEL='3'

# Change to script root
cd $(get_script_dir)
GREEN='\033[0;32m'
NC='\033[0m'

BOLD=$(tput bold)
NORMAL=$(tput sgr0)

# Run style tests
echo -e "${GREEN}${BOLD}Running style tests:${NC}"
flake8 $module --exclude test*.py,__init__.py --show-source

# Ignore import errors for __init__ and tests
flake8 $module --filename=__init__.py,test*.py --ignore=F --show-source

echo -e "${GREEN}${BOLD}Testing docstring conventions:${NC}"
# Test docstring conventions
pydocstyle $module --match='(?!__init__).*\.py' 2>&1 | grep -v "WARNING: __all__"

# Run unit tests python 2.7
echo -e "${GREEN}${BOLD}Running unit tests for 2.7:${NC}"
nosetests-2.7 -v --with-doctest --with-coverage --cover-erase --cover-package=$module $module

echo -e "${GREEN}${BOLD}Running unit tests for 3.5:${NC}"
nosetests-3.5 -v --with-doctest --with-coverage --cover-erase --cover-package=$module $module

# Export html
coverage html
