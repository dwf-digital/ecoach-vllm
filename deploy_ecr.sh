#!/bin/bash

# Set your AWS region, ECR repository name, and image name
AWS_REGION=ap-southeast-1
ECR_REPOSITORY_NAME=training-guru-llm
IMAGE_NAME=training-guru
VERSION_TAG=v1.0.0

# Authenticate Docker to your ECR registry
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com

# Build the Docker image
docker build -t $IMAGE_NAME .

# Tag the Docker image with the ECR repository URI for v1.0.0
ECR_IMAGE_URI=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$VERSION_TAG
docker tag $IMAGE_NAME $ECR_IMAGE_URI

# Push the Docker image to ECR for both tags
docker push $ECR_IMAGE_URI
