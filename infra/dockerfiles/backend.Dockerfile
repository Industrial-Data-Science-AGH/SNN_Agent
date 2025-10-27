FROM python:3.11-slim
WORKDIR /app
COPY ../backend/requirements.txt .
RUN pip install -r requirements.txt
COPY ../backend /app
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
