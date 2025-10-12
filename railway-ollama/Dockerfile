FROM ollama/ollama:latest

# Pre-download llama3.2 model during build
# This ensures the model is always available
RUN ollama serve & \
    sleep 10 && \
    ollama pull llama3.2 && \
    pkill ollama

# Expose Ollama port
EXPOSE 11434

# Start Ollama server
CMD ["ollama", "serve"]
