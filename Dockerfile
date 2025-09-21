FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# System libs for Pillow and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    libfreetype6 \
    libtiff6 \
    libwebp7 \
    libopenjp2-7 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code and assets
COPY . .

EXPOSE 8501

# Streamlit server
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
