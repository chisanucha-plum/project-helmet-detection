cd motorcycle-safety-violation-detection

pip install -r backend/requirements.txt
pip install pytest pytest-asyncio pytest-mock httpx

pytest test/unit/ -v

pytest test/integration/ -v

pip install pytest-cov
pytest test/ --cov=backend/app --cov-report=term-missing