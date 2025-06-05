FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir scripts

COPY scripts/inference.py scripts/

CMD ["python", "scripts/inference.py"]