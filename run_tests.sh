#!/bin/bash

# Change directory to the location of your test files
cd /tests

# Print a message before starting the tests
echo "Running tests..."

# Find test files and folders starting with "test"
while IFS= read -r test_item; do
    if [[ -f "$test_item" ]]; then
        echo "Running tests from file: $test_item"
        pytest "$test_item"
    elif [[ -d "$test_item" ]]; then
        echo "Entering directory: $test_item"
        cd "$test_item"
            for test_file in test_*.py; do
        echo "Running tests from $test_file"
    pytest "$test_file"
done
        cd ..
    fi
done < <(find . -type f -name 'test_*.py' -o -type d -name 'test*')

# Print a message after the tests are finished
echo "Tests completed."
