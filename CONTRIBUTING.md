# Contribution Guidelines

## Setting up the environment
You need to make sure that you have the right setup. Install anaconda, and you can use the following script
to setup your environment:

```
conda create -q -y -n solt_test_env python=3.11 pandoc
source activate solt_test_env
pip install -r ci/requirements.txt
pip install pre-commit
pre-commit install
pip install -e .
```

## Contributing

1. Implement the augmentation
2. Write tests
3. Make sure that your codecoverage is 100%. The criteria for merging into master is
that all of the tests pass and that the codecoverage is 100%.

Before commiting commiting, execute these actions:
```
git add .
black --config=black.toml .  
pre-commit
```

If everything is alright, this will be the expected output:
```
black....................................................................Passed
flake8...................................................................Passed
check for case conflicts.................................................Passed
check for broken symlinks............................(no files to check)Skipped
fix end of files.........................................................Passed
fix requirements.txt.................................(no files to check)Skipped
```
