# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Security: Set a non-root user
    APP_USER=appuser \
    # Security: Set specific version for pip
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create a non-root user and set up permissions
RUN groupadd -r ${APP_USER} && \
    useradd -r -g ${APP_USER} -d /app ${APP_USER} && \
    chown -R ${APP_USER}:${APP_USER} /app

# Install system dependencies with security updates
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    # Security: Add security updates
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    # Security: Clean up apt cache
    && apt-get clean \
    # Security: Remove unnecessary packages
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false

# Copy requirements first to leverage Docker cache
COPY --chown=${APP_USER}:${APP_USER} requirements.txt .

# Install Python dependencies with security checks
RUN pip install --no-cache-dir --upgrade pip
RUN ./install_complete.sh

# Copy application code with proper permissions
COPY --chown=${APP_USER}:${APP_USER} . .

# Security: Switch to non-root user
USER ${APP_USER}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with proper signal handling
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--timeout-keep-alive", "75", "--reload"] 