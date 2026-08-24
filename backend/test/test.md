# Backend tests

Run from `backend/` (activate the venv first):

    pip install -r requirements.txt

    python -m pytest tests/unit -v

    python -m pytest tests/integration -v

    pip install pytest-cov
    python -m pytest --cov=app --cov-report=term-missing

`backend/pytest.ini` sets `testpaths = tests` and `pythonpath = .`, so plain
`python -m pytest` from `backend/` also runs the whole suite.