# Deployment & Operations Guide

## 🚀 Production Deployment

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.5 GHz
- **Memory**: 8 GB RAM
- **Storage**: 50 GB SSD
- **Network**: 1 Gbps connection
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Docker

#### Recommended Production
- **CPU**: 8+ cores, 3.0 GHz
- **Memory**: 16-32 GB RAM
- **Storage**: 500+ GB SSD (NVMe preferred)
- **Network**: 10 Gbps connection
- **GPU**: Optional, for accelerated embeddings

### Environment Setup

#### 1. **Docker Deployment (Recommended)**

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - qdrant
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_data:
```

#### 2. **Kubernetes Deployment**

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: evolutyz/rag-python:latest
        ports:
        - containerPort: 8000
        env:
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: gemini-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Configuration Management

#### Production Environment Variables

```bash
# Core Configuration
QDRANT_URL=http://qdrant-cluster:6333
QDRANT_API=your_production_qdrant_key
GEMINI_API_KEY=your_production_gemini_key
COLLECTION_NAME=production_rag_collection

# Performance Optimization
INGEST_BATCH_SIZE=32  # Reduced for stability
QDRANT_MAX_POINTS_PER_UPSERT=8  # Conservative for production
QDRANT_UPSERT_RETRIES=5  # More retries in production
QDRANT_UPSERT_BACKOFF_BASE=1.0  # Longer backoff

# Memory Management
CSV_CHUNK_TOKENS=150  # Slightly smaller for memory efficiency
PDF_CHUNK_TOKENS=150
CSV_CHUNK_MAX_TOKENS=250
PDF_CHUNK_MAX_TOKENS=250

# Feature Configuration
ENABLE_LOCAL_ROUTING=1  # Enable for better performance
SUMMARY_MAX_TOP_K=20   # Reasonable limit for production
STYLE_DETAILED_MAX_WORDS=150  # Controlled response length

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=1
HEALTH_CHECK_INTERVAL=30
```

#### Security Configuration

```bash
# Security Settings
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
API_KEY_REQUIRED=1  # Enable API key authentication
RATE_LIMIT_REQUESTS=100  # Requests per minute
RATE_LIMIT_WINDOW=60     # Window in seconds

# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/yourdomain.crt
SSL_KEY_PATH=/etc/ssl/private/yourdomain.key
```

## 📊 Monitoring & Observability

### Application Metrics

#### Key Performance Indicators (KPIs)

1. **Response Time Metrics**
   - Average query response time: < 200ms (target)
   - 95th percentile response time: < 500ms
   - 99th percentile response time: < 1000ms

2. **Throughput Metrics**
   - Queries per second (QPS)
   - Documents processed per minute
   - Successful vs failed requests ratio

3. **Quality Metrics**
   - Average relevance score
   - Source diversity index
   - User satisfaction ratings

#### Prometheus Metrics

```python
# Add to your FastAPI app
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('rag_request_duration_seconds', 'Request latency')
ACTIVE_CONNECTIONS = Gauge('rag_active_connections', 'Active connections')

# Business metrics
DOCUMENTS_PROCESSED = Counter('rag_documents_processed_total', 'Documents processed')
QUERY_RESULTS = Histogram('rag_query_results_count', 'Number of results returned')
EMBEDDING_CACHE_HITS = Counter('rag_cache_hits_total', 'Cache hits')
```

#### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "RAG Pipeline Monitoring",
    "panels": [
      {
        "title": "Query Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rag_request_duration_seconds_bucket)",
            "legend": "95th percentile"
          }
        ]
      },
      {
        "title": "Requests per Second",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(rag_requests_total[5m])",
            "legend": "QPS"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

#### Application Health Endpoint

```python
@app.get("/health")
def health_check():
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "database": check_qdrant_connection(),
            "embeddings": check_embedding_models(),
            "llm": check_gemini_connection(),
            "memory": check_memory_usage(),
            "disk": check_disk_space()
        }
    }
    
    # Determine overall status
    if any(check["status"] == "unhealthy" for check in health_status["checks"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

#### Kubernetes Health Checks

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Logging Strategy

#### Structured Logging Configuration

```python
import structlog
import logging

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Usage example
logger = structlog.get_logger()
logger.info("Query processed", 
           query_id="12345",
           response_time_ms=234,
           result_count=5,
           user_id="user_123")
```

#### Log Aggregation with ELK Stack

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  paths:
    - /app/logs/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "rag-logs-%{+yyyy.MM.dd}"

setup.template.settings:
  index.number_of_shards: 1
  index.codec: best_compression
```

## 🔧 Maintenance & Operations

### Backup Strategy

#### Database Backup

```bash
#!/bin/bash
# backup_qdrant.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/qdrant_$DATE"

# Create Qdrant snapshot
curl -X POST "http://qdrant:6333/collections/production_rag_collection/snapshots"

# Copy snapshot files
mkdir -p $BACKUP_DIR
docker cp qdrant:/qdrant/storage/snapshots $BACKUP_DIR/

# Compress and upload to S3
tar -czf "$BACKUP_DIR.tar.gz" -C /backups "qdrant_$DATE"
aws s3 cp "$BACKUP_DIR.tar.gz" s3://your-backup-bucket/qdrant/

# Cleanup local files older than 7 days
find /backups -name "qdrant_*" -mtime +7 -delete
```

#### Configuration Backup

```bash
#!/bin/bash
# backup_config.sh

# Backup environment configuration
kubectl get configmaps -o yaml > configs_backup_$(date +%Y%m%d).yaml
kubectl get secrets -o yaml > secrets_backup_$(date +%Y%m%d).yaml

# Backup deployment manifests
kubectl get deployments -o yaml > deployments_backup_$(date +%Y%m%d).yaml
```

### Update Procedures

#### Rolling Updates

```bash
#!/bin/bash
# rolling_update.sh

# 1. Update Docker image
docker build -t evolutyz/rag-python:v2.0 .
docker push evolutyz/rag-python:v2.0

# 2. Update Kubernetes deployment
kubectl set image deployment/rag-api rag-api=evolutyz/rag-python:v2.0

# 3. Monitor rollout
kubectl rollout status deployment/rag-api

# 4. Verify health
kubectl get pods -l app=rag-api
curl -f http://your-api-url/health || kubectl rollout undo deployment/rag-api
```

#### Database Migration

```python
# migration_script.py
def migrate_collection_schema():
    """Migrate Qdrant collection to new schema version."""
    
    # 1. Create new collection with updated schema
    client.create_collection(
        collection_name="production_rag_collection_v2",
        vectors_config=new_vector_config
    )
    
    # 2. Copy data with transformation
    batch_size = 100
    offset = 0
    
    while True:
        points = client.scroll(
            collection_name="production_rag_collection",
            limit=batch_size,
            offset=offset
        )
        
        if not points[0]:
            break
            
        # Transform points if needed
        transformed_points = [transform_point(p) for p in points[0]]
        
        # Upsert to new collection
        client.upsert(
            collection_name="production_rag_collection_v2",
            points=transformed_points
        )
        
        offset += batch_size
    
    # 3. Switch collection alias
    client.update_collection_aliases(
        actions=[
            AliasOperations(
                create_alias=CreateAlias(
                    collection_name="production_rag_collection_v2",
                    alias_name="production_rag_collection"
                )
            )
        ]
    )
```

### Performance Optimization

#### Database Optimization

```python
# qdrant_optimization.py

# 1. Optimize HNSW parameters
hnsw_config = HnswConfigDiff(
    m=16,  # Number of bi-directional links created for every new element
    ef_construct=200,  # Size of the dynamic candidate list
    full_scan_threshold=10000,  # Threshold for switching to full scan
    max_indexing_threads=4  # Number of parallel threads for indexing
)

# 2. Configure quantization for memory efficiency
quantization_config = ScalarQuantization(
    type=ScalarType.INT8,
    quantile=0.99,
    always_ram=True
)

# 3. Set optimal collection parameters
client.update_collection(
    collection_name="production_rag_collection",
    hnsw_config=hnsw_config,
    quantization_config=quantization_config,
    optimizer_config=OptimizersConfigDiff(
        default_segment_number=4,
        max_segment_size=20000,
        memmap_threshold=20000,
        indexing_threshold=20000,
        flush_interval_sec=30,
        max_optimization_threads=2
    )
)
```

#### Application Optimization

```python
# app_optimization.py

# 1. Connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:pass@localhost/db",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=300
)

# 2. Response caching
from cachetools import TTLCache
import hashlib

query_cache = TTLCache(maxsize=1000, ttl=300)  # 5-minute TTL

def cached_query(query_text: str, top_k: int):
    cache_key = hashlib.md5(f"{query_text}:{top_k}".encode()).hexdigest()
    
    if cache_key in query_cache:
        return query_cache[cache_key]
    
    result = perform_query(query_text, top_k)
    query_cache[cache_key] = result
    return result

# 3. Async processing
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def async_embed_documents(documents):
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, embed_document, doc)
        for doc in documents
    ]
    return await asyncio.gather(*tasks)
```

### Troubleshooting Guide

#### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory consumption
   kubectl top pods
   
   # Reduce batch sizes
   export INGEST_BATCH_SIZE=16
   export QDRANT_MAX_POINTS_PER_UPSERT=4
   
   # Enable memory monitoring
   export MEMORY_THRESHOLD_MB=6000
   ```

2. **Slow Query Performance**
   ```bash
   # Check Qdrant performance
   curl "http://qdrant:6333/collections/production_rag_collection"
   
   # Optimize HNSW parameters
   # Reduce ef_construct if indexing is slow
   # Increase ef if search accuracy is low
   ```

3. **Connection Timeouts**
   ```bash
   # Increase timeouts
   export QDRANT_TIMEOUT=30
   export GEMINI_TIMEOUT=60
   
   # Check network connectivity
   kubectl exec -it pod-name -- ping qdrant
   ```

4. **Embedding Model Loading Issues**
   ```bash
   # Preload models at startup
   export PRELOAD_MODELS=1
   
   # Use model mirroring for reliability
   export MODEL_MIRROR_URL=https://your-model-mirror.com
   ```

#### Debug Mode

```python
# Enable debug logging
export LOG_LEVEL=DEBUG
export QDRANT_DEBUG=1
export EMBEDDING_DEBUG=1

# Additional debugging endpoints
@app.get("/debug/memory")
def debug_memory():
    import psutil
    process = psutil.Process()
    return {
        "memory_percent": process.memory_percent(),
        "memory_info": process.memory_info()._asdict(),
        "cpu_percent": process.cpu_percent()
    }

@app.get("/debug/models")
def debug_models():
    return {
        "dense_model_loaded": hasattr(embedder, 'dense_model'),
        "colbert_model_loaded": hasattr(embedder, 'colbert_model'),
        "sparse_model_loaded": hasattr(embedder, 'sparse_model')
    }
```

This comprehensive deployment and operations guide ensures your RAG pipeline runs reliably in production with proper monitoring, maintenance, and troubleshooting capabilities.