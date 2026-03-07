#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/db/local/docker-compose.yml"
ENV_FILE="$REPO_ROOT/db/local/.env"
ENV_EXAMPLE_FILE="$REPO_ROOT/db/local/.env.example"

if [ ! -f "$ENV_FILE" ]; then
  ENV_FILE="$ENV_EXAMPLE_FILE"
fi

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d postgres
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --profile migrations run --rm flyway migrate
