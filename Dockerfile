# Use a more modern Python image
FROM python:3.12-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip first to avoid issues with new packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and data
COPY src/ src/
COPY data/ data/

CMD ["python", "src/main.py"]