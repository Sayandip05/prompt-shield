#!/usr/bin/env bash
set -e
git pull
docker compose -f docker-compose.prod.yml up -d --build
