FROM python:3.11-slim

# Install build deps for TA-Lib if needed (uncomment if you need to build from source)
# RUN apt-get update && apt-get install -y build-essential wget libbz2-dev liblzma-dev \
#     libssl-dev libffi-dev libta-lib-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["python", "market_master_ai.py"]
