FROM python:3.11-slim

WORKDIR /app/python_rag

COPY python_rag/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY python_rag/ ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
