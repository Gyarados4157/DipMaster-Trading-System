# 🚀 Deployment Guide - DipMaster Trading System

This guide covers all deployment methods for the DipMaster Trading System, from local development to production-ready deployments.

## Table of Contents

- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Setup](#production-setup)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Security Guidelines](#security-guidelines)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.11+ or Docker
- Binance account (for live trading)
- 8GB+ RAM recommended
- Stable internet connection

### 5-Minute Setup

```bash
# Clone repository
git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git
cd DipMaster-Trading-System

# Quick start with Docker
cp config/config.json.example config/config.json
# Edit config.json with your settings
docker-compose up -d

# Check status
docker-compose logs -f dipmaster
```

## Local Development

### Environment Setup

#### 1. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black isort flake8
```

#### 2. Configuration

```bash
# Copy example configuration
cp config/config.json.example config/config.json

# Edit configuration
nano config/config.json
```

**Minimal Configuration:**
```json
{
  "symbols": ["BTCUSDT"],
  "paper_trading": true,
  "max_positions": 1,
  "max_position_size": 100,
  "enable_dashboard": true,
  "log_level": "INFO"
}
```

#### 3. Run Development Mode

```bash
# Start in paper trading mode
python main.py --paper --config config/config.json --log-level DEBUG

# With dashboard disabled (for debugging)
python main.py --paper --config config/config.json --no-dashboard
```

### Development Tools

#### Testing
```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_signal_detector.py -v
```

#### Code Quality
```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/

# Type checking (if using mypy)
mypy src/
```

## Docker Deployment

### Single Container

#### Build Image
```bash
# Build production image
docker build -t dipmaster-trading .

# Build with custom tag
docker build -t dipmaster-trading:v1.0.0 .
```

#### Run Container
```bash
# Basic run
docker run -d \
  --name dipmaster \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  dipmaster-trading

# With environment variables
docker run -d \
  --name dipmaster \
  -e PAPER_TRADING=true \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  dipmaster-trading

# With restart policy
docker run -d \
  --name dipmaster \
  --restart unless-stopped \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  dipmaster-trading
```

### Docker Compose

#### Production Compose File
```yaml
version: '3.8'

services:
  dipmaster:
    build: .
    container_name: dipmaster-trading
    restart: unless-stopped
    environment:
      - LOG_LEVEL=INFO
      - TZ=UTC
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: dipmaster-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    container_name: dipmaster-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

#### Development Compose File
```yaml
version: '3.8'

services:
  dipmaster-dev:
    build: 
      context: .
      dockerfile: Dockerfile.dev
    container_name: dipmaster-dev
    environment:
      - PAPER_TRADING=true
      - LOG_LEVEL=DEBUG
    volumes:
      - .:/app
      - ./config:/app/config
      - ./logs:/app/logs
    ports:
      - "8080:8080"
      - "5678:5678"  # Debug port
    command: ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "main.py", "--config", "config/config.json"]
```

### Docker Management

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f dipmaster

# Stop services
docker-compose down

# Update and restart
docker-compose pull
docker-compose up -d

# Cleanup
docker-compose down -v --remove-orphans
```

## Cloud Deployment

### AWS Deployment

#### ECS (Elastic Container Service)

1. **Build and Push Image**
```bash
# Create ECR repository
aws ecr create-repository --repository-name dipmaster-trading

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Build and tag image
docker build -t dipmaster-trading .
docker tag dipmaster-trading:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/dipmaster-trading:latest

# Push image
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/dipmaster-trading:latest
```

2. **ECS Task Definition**
```json
{
  "family": "dipmaster-trading",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "dipmaster",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/dipmaster-trading:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "secrets": [
        {
          "name": "BINANCE_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:dipmaster/api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/dipmaster-trading",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### EC2 Instance

```bash
# Launch EC2 instance (Ubuntu 20.04+)
# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Deploy application
git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git
cd DipMaster-Trading-System
cp config/config.json.example config/config.json
# Edit config.json

# Start with docker-compose
docker-compose up -d
```

### Google Cloud Platform

#### Cloud Run
```bash
# Enable APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/dipmaster-trading

# Deploy to Cloud Run
gcloud run deploy dipmaster-trading \
  --image gcr.io/PROJECT-ID/dipmaster-trading \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 1 \
  --set-env-vars LOG_LEVEL=INFO
```

### Azure Container Instances

```bash
# Create resource group
az group create --name dipmaster-rg --location eastus

# Deploy container
az container create \
  --resource-group dipmaster-rg \
  --name dipmaster-trading \
  --image dipmaster-trading:latest \
  --cpu 1 \
  --memory 2 \
  --restart-policy Always \
  --environment-variables LOG_LEVEL=INFO \
  --ports 8080
```

## Production Setup

### Environment Configuration

#### Production Config Template
```json
{
  "symbols": [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"
  ],
  "paper_trading": false,
  "api_key": "${BINANCE_API_KEY}",
  "api_secret": "${BINANCE_API_SECRET}",
  "max_positions": 3,
  "max_position_size": 1000,
  "daily_loss_limit": -500,
  "rsi_entry_range": [30, 50],
  "dip_threshold": 0.002,
  "volume_multiplier": 1.5,
  "max_holding_minutes": 180,
  "target_profit": 0.008,
  "min_confidence": 0.6,
  "enable_dashboard": true,
  "dashboard_port": 8080,
  "log_level": "INFO",
  "log_rotation": "daily",
  "backup_enabled": true,
  "health_check_port": 8081
}
```

#### Environment Variables

```bash
# Create .env file (never commit to git)
cat > .env << EOF
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
PAPER_TRADING=false
LOG_LEVEL=INFO
MAX_POSITIONS=3
MAX_POSITION_SIZE=1000
DAILY_LOSS_LIMIT=-500
EOF
```

### High Availability Setup

#### Load Balancer Configuration
```nginx
upstream dipmaster_backend {
    server dipmaster-1:8080;
    server dipmaster-2:8080 backup;
}

server {
    listen 80;
    server_name dipmaster.yourdomain.com;

    location / {
        proxy_pass http://dipmaster_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /health {
        proxy_pass http://dipmaster_backend/health;
        access_log off;
    }
}
```

#### Database Setup (Optional)
```yaml
# Add to docker-compose.yml
services:
  postgres:
    image: postgres:13
    container_name: dipmaster-db
    environment:
      POSTGRES_DB: dipmaster
      POSTGRES_USER: dipmaster
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    container_name: dipmaster-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Monitoring & Maintenance

### Health Checks

```bash
# Built-in health check
curl http://localhost:8080/health

# System status
curl http://localhost:8080/status

# Metrics endpoint
curl http://localhost:8080/metrics
```

### Logging Setup

#### Log Rotation
```bash
# Setup logrotate
sudo tee /etc/logrotate.d/dipmaster << EOF
/opt/dipmaster/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 dipmaster dipmaster
    postrotate
        docker kill -s HUP dipmaster 2>/dev/null || true
    endscript
}
EOF
```

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/backup/dipmaster/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r config/ "$BACKUP_DIR/"

# Backup logs
cp -r logs/ "$BACKUP_DIR/"

# Backup data
cp -r data/ "$BACKUP_DIR/"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR.tar.gz" s3://your-backup-bucket/dipmaster/

# Cleanup old backups (keep 30 days)
find /backup/dipmaster/ -name "*.tar.gz" -mtime +30 -delete
```

### Monitoring Script

```bash
#!/bin/bash
# monitor.sh - Health monitoring script

WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

check_health() {
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
    
    if [ "$response" != "200" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"🚨 DipMaster Trading System is down! HTTP: '$response'"}' \
            "$WEBHOOK_URL"
        return 1
    fi
    return 0
}

check_memory() {
    memory_usage=$(docker stats --no-stream dipmaster | awk 'NR==2 {print $4}' | sed 's/%//')
    
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"⚠️ DipMaster high memory usage: '$memory_usage'%"}' \
            "$WEBHOOK_URL"
    fi
}

# Run checks
check_health
check_memory

# Add to crontab: */5 * * * * /opt/dipmaster/monitor.sh
```

## Security Guidelines

### API Security

1. **API Key Management**
```bash
# Use environment variables, never hardcode
export BINANCE_API_KEY="your_key_here"
export BINANCE_API_SECRET="your_secret_here"

# Or use Docker secrets
echo "your_api_key" | docker secret create binance_api_key -
echo "your_secret" | docker secret create binance_api_secret -
```

2. **Binance API Restrictions**
```
- Enable IP whitelist
- Disable withdrawals
- Enable only spot trading
- Set daily limits
```

### Network Security

```bash
# Firewall rules (ufw)
sudo ufw allow ssh
sudo ufw allow 8080/tcp  # Dashboard
sudo ufw deny 5432/tcp   # Database (if used)
sudo ufw enable
```

### Container Security

```dockerfile
# Use non-root user in Dockerfile
RUN addgroup --system --gid 1001 dipmaster
RUN adduser --system --uid 1001 dipmaster
USER dipmaster
```

## Troubleshooting

### Common Issues

#### 1. WebSocket Connection Failed
```bash
# Check network connectivity
ping stream.binance.com

# Check firewall
sudo ufw status

# Restart container
docker-compose restart dipmaster
```

#### 2. High Memory Usage
```bash
# Check container stats
docker stats dipmaster

# Restart with memory limit
docker run --memory="2g" dipmaster-trading

# Check logs for memory leaks
docker logs dipmaster | grep -i memory
```

#### 3. API Authentication Failed
```bash
# Verify API keys
python -c "
import os
from binance.client import Client
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
print(client.get_account())
"

# Check API permissions
curl -X GET 'https://api.binance.com/api/v3/account' -H 'X-MBX-APIKEY: YOUR_API_KEY'
```

#### 4. Database Connection Issues
```bash
# Test database connection
docker exec -it dipmaster-db psql -U dipmaster -d dipmaster -c "SELECT 1;"

# Reset database
docker-compose down -v
docker-compose up -d postgres
# Wait for postgres to start
docker-compose up -d dipmaster
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug output
python main.py --config config/config.json --log-level DEBUG --no-dashboard

# Attach debugger
python -m pdb main.py
```

### Performance Optimization

```bash
# Monitor performance
docker stats dipmaster

# Optimize memory usage
# In docker-compose.yml:
services:
  dipmaster:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

## Support

### Getting Help

1. **Check Logs**: Always start with `docker logs dipmaster`
2. **Review Configuration**: Validate your `config.json`
3. **Test Components**: Use the health check endpoints
4. **Monitor Resources**: Check CPU, memory, and network usage

### Reporting Issues

When reporting deployment issues, please include:
- Deployment method (Docker, native, cloud)
- Configuration (sanitized, no API keys)
- Log excerpts (last 50-100 lines)
- System specifications
- Error messages

---

**Last Updated**: 2025-08-12  
**Deployment Guide Version**: 1.0.0  
**Compatible with**: DipMaster Trading System v1.0.0