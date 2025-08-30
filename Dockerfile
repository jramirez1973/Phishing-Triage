COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Copy frontend files
COPY --chown=appuser:appuser frontend /app/frontend

# Create necessary directories
RUN mkdir -p data storage ml/metrics logs && \
    chown -R appuser:appuser /app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
