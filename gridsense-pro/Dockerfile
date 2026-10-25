# Works on Hugging Face Spaces (Docker), Railway, Fly.io, or any Docker host
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860 DATA_DIR=/tmp/gridsense PYTHONUNBUFFERED=1
EXPOSE 7860
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
