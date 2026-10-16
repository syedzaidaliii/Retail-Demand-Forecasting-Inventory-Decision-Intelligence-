# syntax=docker/dockerfile:1
# One Dockerfile, two images:  docker build --target api .   |   docker build --target dashboard .

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app
WORKDIR /app

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY configs ./configs
COPY artifacts ./artifacts
RUN useradd --create-home appuser
USER appuser

# ---------------------------------------------------------------- API
FROM base AS api
EXPOSE 8000
# The serving data is not baked into the image: mount ./data to /app/data (see docker-compose.yml)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- dashboard
FROM base AS dashboard
USER root
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt
COPY dashboard ./dashboard
COPY reports ./reports
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
