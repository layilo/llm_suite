FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "llm_benchmark_suite.cli", "demo", "--output-dir", "artifacts/generated/docker-demo"]
