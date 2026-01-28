#!/bin/bash
# Build and run Triton + FastAPI services

set -e

PROJECT_ROOT=$(pwd)

echo "=================================================="
echo "Building DSA Agent with Triton Integration"
echo "=================================================="

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Build images
echo ""
echo "Building Docker images..."
docker-compose build

echo ""
echo "=================================================="
echo "Build complete! Starting services..."
echo "=================================================="
echo ""
echo "Services will be available at:"
echo "  - FastAPI (REST API): http://localhost:8000"
echo "  - Triton HTTP: http://localhost:8001"
echo "  - Triton gRPC: localhost:8002"
echo "  - Triton Metrics: http://localhost:8003"
echo ""
echo "Starting containers..."
echo ""

docker-compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 10

# Health check
echo ""
echo "Health Status:"
echo "=============="

# Check FastAPI
echo -n "FastAPI Server: "
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Running"
else
    echo "✗ Not responding"
fi

# Check Triton
echo -n "Triton Server: "
if curl -s http://localhost:8001/v2/health/live > /dev/null 2>&1; then
    echo "✓ Running"
else
    echo "✗ Not responding"
fi

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
