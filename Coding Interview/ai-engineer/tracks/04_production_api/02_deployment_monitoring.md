# Challenge 02: Deployment & Monitoring (Production)

**Difficulty**: Advanced
**Time Estimate**: 50-60 minutes
**Interview Section**: Production Deployment

---

## 📋 Challenge Description

Prepare your RAG API for **production deployment** with:
1. Health checks and readiness probes
2. Metrics collection (Prometheus format)
3. Rate limiting
4. CORS configuration
5. Docker containerization

---

## 🎯 Requirements

### Part A: Health Checks (10 min)

```python
from fastapi import status

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Basic health check endpoint.
    Used by load balancers to check if service is alive.
    """
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """
    Readiness check - verify all dependencies are ready.
    Check database, cache, vector store, etc.
    """
    checks = {
        "cache": await check_cache(),
        "vector_db": await check_vector_db(),
        "llm": await check_llm()
    }

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks
    }
```

### Part B: Metrics Collection (15 min)

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
query_counter = Counter(
    'rag_queries_total',
    'Total number of RAG queries',
    ['status']  # labels: success, error, cached
)

query_latency = Histogram(
    'rag_query_duration_seconds',
    'RAG query latency in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

cache_hit_counter = Counter(
    'rag_cache_hits_total',
    'Total number of cache hits'
)

@app.get("/metrics")
async def metrics():
    """Expose metrics in Prometheus format"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    # Track metrics
    with query_latency.time():
        try:
            result = await process_query(request)
            query_counter.labels(status='success').inc()
            if result["cached"]:
                cache_hit_counter.inc()
            return result
        except Exception as e:
            query_counter.labels(status='error').inc()
            raise
```

### Part C: Rate Limiting (10 min)

```python
from fastapi import Request, HTTPException
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < 60
        ]

        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False

        # Record request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(requests_per_minute=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host

    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": 60}
        )

    return await call_next(request)
```

### Part D: CORS Configuration (5 min)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # In production
    # allow_origins=["*"],  # Only for development!
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Part E: Docker Deployment (15 min)

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## 📊 Monitoring Dashboard

### Prometheus Queries

```promql
# Query rate (per minute)
rate(rag_queries_total[1m])

# Error rate
rate(rag_queries_total{status="error"}[5m])

# 95th percentile latency
histogram_quantile(0.95, rag_query_duration_seconds_bucket)

# Cache hit rate
rate(rag_cache_hits_total[5m]) / rate(rag_queries_total[5m]) * 100
```

### Grafana Dashboard Panels

1. **Query Rate**: Line graph of queries per second
2. **Error Rate**: Line graph of errors per second
3. **Latency**: Histogram of query latencies (p50, p95, p99)
4. **Cache Hit Rate**: Gauge showing percentage
5. **Active Sessions**: Counter of unique session IDs
6. **Top Queries**: Table of most frequent questions

---

## 💡 Deployment Checklist

### Before Production

- [ ] Environment variables configured (API keys, secrets)
- [ ] Logging level set to INFO or WARNING
- [ ] Error tracking enabled (Sentry, etc.)
- [ ] Rate limiting configured
- [ ] CORS properly restricted
- [ ] SSL/TLS certificates installed
- [ ] Database connections pooled
- [ ] Secrets not in code (use env vars or secrets manager)
- [ ] Health checks tested
- [ ] Load testing completed
- [ ] Monitoring dashboards created
- [ ] Alerts configured

### Docker Commands

```bash
# Build image
docker build -t rag-api:latest .

# Run container
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  rag-api:latest

# View logs
docker logs -f rag-api

# Check health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics

# Stop container
docker stop rag-api
```

---

## 🎓 Key Concepts

1. **Health Checks**: Liveness vs Readiness probes
2. **Metrics**: Prometheus format (counters, gauges, histograms)
3. **Rate Limiting**: Prevent abuse and DDoS
4. **CORS**: Cross-Origin Resource Sharing security
5. **Docker**: Containerization for consistent deployment
6. **Observability**: Logs, metrics, traces

---

## 🧪 Load Testing

```python
import asyncio
import httpx

async def load_test(num_requests: int = 1000):
    """Send concurrent requests to test API performance"""
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(num_requests):
            task = client.post(
                "http://localhost:8000/query",
                json={"question": f"Test question {i}"}
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        success = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
        errors = num_requests - success

        print(f"Success: {success}/{num_requests}")
        print(f"Errors: {errors}")
        print(f"Success rate: {success/num_requests*100:.2f}%")

# Run
asyncio.run(load_test(1000))
```

---

**Time Allocation**:
- Health checks: 10 min
- Metrics: 15 min
- Rate limiting: 10 min
- CORS: 5 min
- Docker: 15 min
- Testing: 5 min
- **Total**: 60 min

**Good luck!** 🎯
