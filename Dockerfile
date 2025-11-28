FROM python:3.11-slim

# Install Node.js for building frontend
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and build frontend
RUN git clone --depth 1 --branch claude/paint-by-number-setup-01RDLkxBUEAGcBRh5awYyHbx https://github.com/basthros/paintbynumber-app.git /tmp/frontend && \
    cd /tmp/frontend && \
    npm install && \
    npm run build && \
    mkdir -p /app/dist && \
    cp -r dist/* /app/dist/

# Copy backend code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
