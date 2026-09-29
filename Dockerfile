FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models/production ./models/production
ENV PYTHONPATH=/app/src ML_HOME=/app
EXPOSE 8000
CMD ["uvicorn", "mlpipeline.serve:app", "--host", "0.0.0.0", "--port", "8000"]
