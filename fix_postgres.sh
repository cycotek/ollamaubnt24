#!/bin/bash
# -----------------------------------------------------------
# PostgreSQL Fix Script
# Completes DB setup assuming agent_user already exists.
# -----------------------------------------------------------

set -euo pipefail

DB_USER="thecloven"
DB_PASS="Maytheforcebewithmealway5!" # !!! MATCHES THE ONE USED IN setup_server.sh !!!
DB_NAME="aiserver_db"
PG_VERSION=$(pg_config --version 2>/dev/null | awk '{print $2}' | cut -d. -f1 || echo 16) 

echo "--- 1. Set Password for Existing User ---"
# Sets the password for the existing user to ensure it matches the script's expectation
sudo -u postgres psql -c "ALTER USER $DB_USER WITH PASSWORD '$DB_PASS';"
echo "✅ Password for '$DB_USER' set/verified."

echo "--- 2. Create Database (If Not Exists) ---"
# Create the database only if it does not already exist
if ! sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    sudo -u postgres createdb $DB_NAME
    echo "✅ Database '$DB_NAME' created."
else
    echo "☑️ Database '$DB_NAME' already exists. Skipping creation."
fi

echo "--- 3. Install and Enable pgvector ---"
# Ensure pgvector is installed
echo "Attempting to install postgresql-$PG_VERSION-pgvector..."
sudo apt install -y postgresql-"$PG_VERSION"-pgvector

# Enable pgvector extension in the database
echo "Enabling pgvector extension..."
sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"
echo "✅ pgvector extension is enabled."

echo "--- FIX COMPLETE ---"
