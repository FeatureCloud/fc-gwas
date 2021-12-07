#!/bin/bash

echo "Building docker image..."
docker build . --tag featurecloud.ai/gwas_chi_squared
