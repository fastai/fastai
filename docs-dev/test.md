# Testing fastai

## Notebook integration tests

Currently we have few automated tests, so most testing is through integration tests done in Jupyter Notebooks. The two places you should check for notebooks to test your code with are:

 - [The fastai examples](https://github.com/fastai/fastai/tree/master/examples)
 - [The fastai_docs notebooks](https://github.com/fastai/fastai_docs/tree/master/docs_src)

In each case, look for notebooks that have names starting with the application you're working on - e.g. 'text' or 'vision'.


## Automated tests

At the moment there are only a few automated tests, so we need to start expanding it! It's not easy to properly automatically test ML code, but there's lots of opportunities for unit tests. We use [pytest](https://docs.pytest.org/en/latest/). To run the tests:


   ```
   make test
   ```

or


   ```
   python setup.py test
   ```

To run an individual test file:

   ```
   pytest tests/test_core.py
   ```

# Writing Tests

XXX: Needs to be written
