FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and data
COPY src/ src/
COPY data/ data/

# To find modules in 'src/'
ENV PYTHONPATH="/app/src"

CMD ["python", "src/main.py"]