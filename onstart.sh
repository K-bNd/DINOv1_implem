curl -sSL https://install.python-poetry.org | python3 -

export PATH="/root/.local/bin:$PATH"

poetry install --no-root

poetry run python main.py