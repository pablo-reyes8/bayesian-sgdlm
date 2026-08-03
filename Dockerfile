FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system sgdlm && useradd --system --gid sgdlm --create-home sgdlm

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY frontend ./frontend
COPY .streamlit ./.streamlit
RUN python -m pip install --upgrade pip && python -m pip install ".[api,excel,frontend]"

USER sgdlm
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "sgdlm.api:app", "--host", "0.0.0.0", "--port", "8000"]
