#!/bin/bash

# FreelanceFlow - Setup Script for New Features
# This script sets up all the new security fixes and features

echo "🚀 FreelanceFlow - Setting up new features..."
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "Please activate your virtual environment first:"
    echo "  source venv/bin/activate  (Linux/Mac)"
    echo "  venv\\Scripts\\activate  (Windows)"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"
echo ""

# Create migrations
echo "📦 Creating database migrations..."
python manage.py makemigrations
echo ""

# Run migrations
echo "🔄 Running database migrations..."
python manage.py migrate
echo ""

# Create email templates directory
echo "📧 Creating email templates directory..."
mkdir -p templates/emails
echo "✅ Email templates directory created"
echo ""

# Check environment variables
echo "🔍 Checking environment variables..."
if [ -f .env ]; then
    echo "✅ .env file found"
    
    # Check for required variables
    if ! grep -q "FRONTEND_URL" .env; then
        echo "⚠️  FRONTEND_URL not found in .env"
        echo "Adding FRONTEND_URL=http://localhost:3000"
        echo "FRONTEND_URL=http://localhost:3000" >> .env
    fi
    
    if ! grep -q "EMAIL_HOST" .env; then
        echo "⚠️  Email configuration not found in .env"
        echo "Please add email configuration to .env:"
        echo "  EMAIL_HOST=smtp.gmail.com"
        echo "  EMAIL_PORT=587"
        echo "  EMAIL_HOST_USER=your-email@gmail.com"
        echo "  EMAIL_HOST_PASSWORD=your-app-password"
    fi
else
    echo "⚠️  .env file not found"
    echo "Please create .env file from .env.example"
fi
echo ""

# Collect static files (if needed)
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput --clear
echo ""

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update .env with FRONTEND_URL and email configuration"
echo "2. Create email templates in templates/emails/"
echo "3. Run tests: python manage.py test"
echo "4. Start server: python manage.py runserver"
echo ""
echo "📚 Documentation:"
echo "- Security fixes: SECURITY_FIXES_APPLIED.md"
echo "- Complete report: COMPLETE_IMPLEMENTATION_REPORT.md"
echo ""
echo "🎉 All done! Happy coding!"
