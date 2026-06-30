FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY neuralforge/ neuralforge/

RUN pip install --no-cache-dir ".[all]"

EXPOSE 8080

CMD ["neuralforge", "serve", "--host", "0.0.0.0", "--port", "8080"]
