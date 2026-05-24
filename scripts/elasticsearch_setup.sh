#!/bin/bash

# ============================================
# FreelanceFlow - Elasticsearch Setup Script
# ============================================
# This script helps manage Elasticsearch indexes
# for the FreelanceFlow application
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
SERVICE_NAME="web"

echo -e "${GREEN}FreelanceFlow - Elasticsearch Management${NC}"
echo "=========================================="
echo ""

# Function to check if Elasticsearch is healthy
check_elasticsearch() {
    echo -e "${YELLOW}Checking Elasticsearch health...${NC}"
    
    if docker-compose -f "$COMPOSE_FILE" exec elastic curl -f http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=30s > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Elasticsearch is healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Elasticsearch is not healthy${NC}"
        return 1
    fi
}

# Function to create indexes
create_indexes() {
    echo -e "${YELLOW}Creating Elasticsearch indexes...${NC}"
    docker-compose -f "$COMPOSE_FILE" exec "$SERVICE_NAME" python manage.py search_index --create
    echo -e "${GREEN}✓ Indexes created${NC}"
}

# Function to delete indexes
delete_indexes() {
    echo -e "${YELLOW}Deleting Elasticsearch indexes...${NC}"
    docker-compose -f "$COMPOSE_FILE" exec "$SERVICE_NAME" python manage.py search_index --delete -f
    echo -e "${GREEN}✓ Indexes deleted${NC}"
}

# Function to populate indexes
populate_indexes() {
    echo -e "${YELLOW}Populating Elasticsearch indexes...${NC}"
    docker-compose -f "$COMPOSE_FILE" exec "$SERVICE_NAME" python manage.py search_index --populate
    echo -e "${GREEN}✓ Indexes populated${NC}"
}

# Function to rebuild indexes
rebuild_indexes() {
    echo -e "${YELLOW}Rebuilding Elasticsearch indexes...${NC}"
    docker-compose -f "$COMPOSE_FILE" exec "$SERVICE_NAME" python manage.py search_index --rebuild -f
    echo -e "${GREEN}✓ Indexes rebuilt${NC}"
}

# Function to show index stats
show_stats() {
    echo -e "${YELLOW}Elasticsearch Index Statistics:${NC}"
    echo ""
    
    echo "Projects Index:"
    docker-compose -f "$COMPOSE_FILE" exec elastic curl -s http://localhost:9200/projects/_count | python -m json.tool
    echo ""
    
    echo "Freelancers Index:"
    docker-compose -f "$COMPOSE_FILE" exec elastic curl -s http://localhost:9200/freelancers/_count | python -m json.tool
    echo ""
}

# Main menu
case "${1:-}" in
    check)
        check_elasticsearch
        ;;
    create)
        check_elasticsearch && create_indexes
        ;;
    delete)
        check_elasticsearch && delete_indexes
        ;;
    populate)
        check_elasticsearch && populate_indexes
        ;;
    rebuild)
        check_elasticsearch && rebuild_indexes
        ;;
    stats)
        check_elasticsearch && show_stats
        ;;
    setup)
        echo -e "${YELLOW}Running full Elasticsearch setup...${NC}"
        check_elasticsearch
        create_indexes
        populate_indexes
        show_stats
        echo -e "${GREEN}✓ Setup complete!${NC}"
        ;;
    *)
        echo "Usage: $0 {check|create|delete|populate|rebuild|stats|setup}"
        echo ""
        echo "Commands:"
        echo "  check     - Check Elasticsearch health"
        echo "  create    - Create indexes"
        echo "  delete    - Delete indexes"
        echo "  populate  - Populate indexes with data"
        echo "  rebuild   - Delete and recreate indexes with data"
        echo "  stats     - Show index statistics"
        echo "  setup     - Run full setup (create + populate)"
        echo ""
        echo "Environment variables:"
        echo "  COMPOSE_FILE - Docker compose file to use (default: docker-compose.yml)"
        echo ""
        echo "Examples:"
        echo "  $0 setup"
        echo "  $0 rebuild"
        echo "  COMPOSE_FILE=docker-compose.prod.yml $0 setup"
        exit 1
        ;;
esac
