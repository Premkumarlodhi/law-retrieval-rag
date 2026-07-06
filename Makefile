.PHONY: install format test run clean

# Install all dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Format code to PEP 8 standards using black
format:
	black src/ tests/ main.py

# Run unit tests
test:
	pytest tests/ -v

# Run the Phase 1 Pipeline
run:
	python main.py

# Clean up python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete