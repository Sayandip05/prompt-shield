#!/bin/bash
# FreelanceFlow Nginx Setup Script
# Automates Nginx installation and configuration

set -e

echo "🚀 FreelanceFlow Nginx Setup"
echo "=============================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Please run as root (use sudo)${NC}"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo -e "${RED}❌ Cannot detect OS${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Detected OS: $OS${NC}"

# Install Nginx
echo ""
echo "📦 Installing Nginx..."
if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    apt-get update
    apt-get install -y nginx
elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ]; then
    yum install -y nginx
else
    echo -e "${RED}❌ Unsupported OS${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Nginx installed${NC}"

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p /etc/nginx/ssl
mkdir -p /etc/nginx/sites-available
mkdir -p /etc/nginx/sites-enabled
mkdir -p /var/www/freelanceflow/staticfiles
mkdir -p /var/www/freelanceflow/media
mkdir -p /var/cache/nginx/static
mkdir -p /var/log/nginx

echo -e "${GREEN}✓ Directories created${NC}"

# Copy configuration files
echo ""
echo "📝 Copying configuration files..."

# Main nginx.conf
if [ -f "deployment/nginx/nginx.conf" ]; then
    cp deployment/nginx/nginx.conf /etc/nginx/nginx.conf
    echo -e "${GREEN}✓ Main nginx.conf copied${NC}"
else
    echo -e "${YELLOW}⚠ nginx.conf not found, skipping${NC}"
fi

# SSL parameters
if [ -f "deployment/nginx/ssl-params.conf" ]; then
    cp deployment/nginx/ssl-params.conf /etc/nginx/ssl-params.conf
    echo -e "${GREEN}✓ SSL parameters copied${NC}"
else
    echo -e "${YELLOW}⚠ ssl-params.conf not found, skipping${NC}"
fi

# Site configuration
if [ -f "deployment/nginx/freelanceflow.conf" ]; then
    cp deployment/nginx/freelanceflow.conf /etc/nginx/sites-available/freelanceflow.conf
    echo -e "${GREEN}✓ Site configuration copied${NC}"
else
    echo -e "${RED}❌ freelanceflow.conf not found${NC}"
    exit 1
fi

# Enable site
if [ ! -L /etc/nginx/sites-enabled/freelanceflow.conf ]; then
    ln -s /etc/nginx/sites-available/freelanceflow.conf /etc/nginx/sites-enabled/
    echo -e "${GREEN}✓ Site enabled${NC}"
fi

# Remove default site
if [ -L /etc/nginx/sites-enabled/default ]; then
    rm /etc/nginx/sites-enabled/default
    echo -e "${GREEN}✓ Default site removed${NC}"
fi

# Generate DH parameters
echo ""
echo "🔐 Generating Diffie-Hellman parameters (this may take a while)..."
if [ ! -f /etc/nginx/ssl/dhparam.pem ]; then
    openssl dhparam -out /etc/nginx/ssl/dhparam.pem 2048
    echo -e "${GREEN}✓ DH parameters generated${NC}"
else
    echo -e "${YELLOW}⚠ DH parameters already exist, skipping${NC}"
fi

# Test configuration
echo ""
echo "🧪 Testing Nginx configuration..."
if nginx -t; then
    echo -e "${GREEN}✓ Configuration test passed${NC}"
else
    echo -e "${RED}❌ Configuration test failed${NC}"
    exit 1
fi

# Set permissions
echo ""
echo "🔒 Setting permissions..."
chown -R www-data:www-data /var/www/freelanceflow
chmod -R 755 /var/www/freelanceflow
echo -e "${GREEN}✓ Permissions set${NC}"

# Enable and start Nginx
echo ""
echo "🚀 Starting Nginx..."
systemctl enable nginx
systemctl restart nginx

if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✓ Nginx is running${NC}"
else
    echo -e "${RED}❌ Nginx failed to start${NC}"
    systemctl status nginx
    exit 1
fi

# Install Certbot (optional)
echo ""
read -p "📜 Install Certbot for Let's Encrypt SSL? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Installing Certbot..."
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        apt-get install -y certbot python3-certbot-nginx
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ]; then
        yum install -y certbot python3-certbot-nginx
    fi
    echo -e "${GREEN}✓ Certbot installed${NC}"
    
    echo ""
    echo "To obtain SSL certificate, run:"
    echo "  sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com"
fi

# Firewall configuration
echo ""
read -p "🔥 Configure firewall (UFW)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔥 Configuring firewall..."
    if command -v ufw &> /dev/null; then
        ufw allow 'Nginx Full'
        ufw allow OpenSSH
        echo -e "${GREEN}✓ Firewall configured${NC}"
    else
        echo -e "${YELLOW}⚠ UFW not found, skipping${NC}"
    fi
fi

# Summary
echo ""
echo "=============================="
echo -e "${GREEN}✅ Nginx setup complete!${NC}"
echo "=============================="
echo ""
echo "📋 Next steps:"
echo "  1. Update domain in /etc/nginx/sites-available/freelanceflow.conf"
echo "  2. Obtain SSL certificate: sudo certbot --nginx -d yourdomain.com"
echo "  3. Test your site: curl -I http://localhost"
echo "  4. Check logs: sudo tail -f /var/log/nginx/freelanceflow_error.log"
echo ""
echo "📚 Documentation: deployment/nginx/README.md"
echo ""
