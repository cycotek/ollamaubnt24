#!/bin/bash
# wait-for-postgres.sh: Waits for the PostgreSQL service to be ready.

set -e

HOST="$1"
PORT="$2"
TIMEOUT=60
start_time=$(date +%s)

echo "Waiting for PostgreSQL service on $HOST:$PORT..."
until nc -z "$HOST" "$PORT"; do
  current_time=$(date +%s)
  elapsed_time=$((current_time - start_time))
  if [ "$elapsed_time" -ge "$TIMEOUT" ]; then
    echo "Error: PostgreSQL service did not become available within $TIMEOUT seconds."
    exit 1
  fi
  sleep 1
done

echo "PostgreSQL is up and running. Starting FastAPI..."
shift 2
exec "$@"
