FROM python:3.11-slim

# Install Node.js
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install
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

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
