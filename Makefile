# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
FLASK_APP = app.py

# Targets
.PHONY: install run clean

# Create a virtual environment and install dependencies
install:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Run the Flask app
run:
	$(PYTHON) $(FLASK_APP)

# Clean up the environment
clean:
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
