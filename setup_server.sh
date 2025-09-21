#!/bin/bash
# -----------------------------------------------------------
# AI Server Setup Script (Ubuntu 24.04)
# Installs Docker, NVIDIA Container Toolkit, PostgreSQL, and pgvector.
# Designed for the EVGA z390 + RTX 2080 Super setup.
# -----------------------------------------------------------

set -euo pipefail

DB_USER="thecloven"
DB_PASS="Maytheforcebewithmealway5!" # !!! CHANGE THIS !!!
DB_NAME="aiserver_db"
# Use pg_config to find the installed PostgreSQL version dynamically
PG_VERSION=$(pg_config --version 2>/dev/null | awk '{print $2}' | cut -d. -f1 || echo 16) 

echo "--- 1. System Update and Dependency Install ---"
sudo apt update
sudo apt install -y curl wget git postgresql postgresql-contrib build-essential libpq-dev

echo "--- 2. Docker Engine Installation ---"
# Add Docker's official GPG key and repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "--- 3. NVIDIA Container Toolkit (for GPU access) ---"
# Add NVIDIA's official GPG key and repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Add current user to the docker group
sudo usermod -aG docker "$USER"
echo "✅ Docker and NVIDIA setup complete. You must RE-LOGIN (or run 'newgrp docker') for Docker group changes to take effect."

echo "--- 4. PostgreSQL & pgvector Setup on Host ---"
# 4a. Create DB User and Database
sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
sudo -u postgres createdb $DB_NAME

echo "--- 4b. Install pgvector ---"
# Install pgvector package for the detected PostgreSQL version
sudo apt install -y postgresql-"$PG_VERSION"-pgvector

# Enable pgvector extension in the new database
sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "--- SCRIPT COMPLETE ---"
echo "Remember to RE-LOGIN now, then run the project structure script."
