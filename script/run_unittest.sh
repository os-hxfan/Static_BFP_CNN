#!/bin/bash
MY_MESSAGE="Running the python unitest"
echo $MY_MESSAGE
TESTS=$(find ./tests/ -name "*tests.py")
echo $TESTS

for TEST in $TESTS
do
  TEST=${TEST:2} # delete the first 2
  TEST=${TEST////.} # replace / with .
  TEST=${TEST%.py}
  #echo ${TEST}
  echo $(python -m ${TEST})
done
MY_MESSAGE="Finish the python unittest"
echo $MY_MESSAGE
