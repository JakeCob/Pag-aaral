# Deploy Ollama to Railway with Pre-loaded Model

## 🚀 Quick Deploy

1. **Create a new Railway service from this directory:**
   ```bash
   cd railway-ollama
   railway up
   ```

2. **Generate a public domain** in Railway dashboard:
   - Go to your service → Settings → Networking
   - Click "Generate Domain"

3. **Your Ollama is ready!** The llama3.2 model is pre-downloaded.

## 📋 What This Does

- Deploys Ollama to Railway
- Pre-downloads the llama3.2 model (~2GB)
- Exposes port 11434
- Model persists across deployments

## 🧪 Test Your Deployment

```bash
# Check if Ollama is running
curl https://your-domain.railway.app/api/tags

# Test with a prompt
curl https://your-domain.railway.app/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Hello!"
}'
```

## 🔄 To Add More Models

Edit the Dockerfile and add more `ollama pull` commands:

```dockerfile
RUN ollama serve & \
    sleep 10 && \
    ollama pull llama3.2 && \
    ollama pull mistral && \
    pkill ollama
```

## 💾 Storage Note

Models are stored in the Docker image, so they persist across restarts. Railway provides persistent storage automatically.
