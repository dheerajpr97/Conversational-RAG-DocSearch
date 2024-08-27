# First stage: Build the application and dependencies
FROM python:3.10-slim-bookworm AS builder

# Set the working directory
WORKDIR /app

# Install build dependencies and other necessary tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Second stage: Create the final runtime image
FROM python:3.10-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy the dependencies and application files from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Install runtime dependencies (if any)
RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set environment variables
ENV API_KEY=${API_KEY}
ENV IN_DOCKER=1

# Run the application
CMD ["streamlit", "run", "src/main.py"]
