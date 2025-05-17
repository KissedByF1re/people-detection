#!/bin/bash

# Script to run performance tests with CPU limits in Docker

# Build the Docker image
echo "Building Docker image for performance testing..."
docker build -t people-detection-performance -f tests/Dockerfile.performance .

# Run the container with CPU limits (8 CPUs)
echo "Running performance tests with 8 CPU limit..."
docker run --cpus="8" \
  -v "$(pwd)/performance_results:/app/performance_results" \
  people-detection-performance

echo "Performance tests completed. Results are available in the performance_results directory." 