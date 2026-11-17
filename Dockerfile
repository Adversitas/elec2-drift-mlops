# SageMaker-compatible inference image: SageMaker runs `docker run <image> serve`
# and expects the model in /opt/ml/model and the server on port 8080.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["driftops"]
CMD ["serve", "--model-dir", "/opt/ml/model", "--capture", "/tmp/capture.jsonl"]
