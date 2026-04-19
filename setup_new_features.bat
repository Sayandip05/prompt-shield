@echo off
REM FreelanceFlow - Setup Script for New Features (Windows)
REM This script sets up all the new security fixes and features

echo.
echo 🚀 FreelanceFlow - Setting up new features...
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️  Warning: Virtual environment not activated
    echo Please activate your virtual environment first:
    echo   venv\Scripts\activate
    exit /b 1
)

echo ✅ Virtual environment detected: %VIRTUAL_ENV%
echo.

REM Create migrations
echo 📦 Creating database migrations...
python manage.py makemigrations
echo.

REM Run migrations
echo 🔄 Running database migrations...
python manage.py migrate
echo.

REM Create email templates directory
echo 📧 Creating email templates directory...
if not exist "templates\emails" mkdir templates\emails
echo ✅ Email templates directory created
echo.

REM Check environment variables
echo 🔍 Checking environment variables...
if exist .env (
    echo ✅ .env file found
    
    REM Check for required variables
    findstr /C:"FRONTEND_URL" .env >nul
    if errorlevel 1 (
        echo ⚠️  FRONTEND_URL not found in .env
        echo Adding FRONTEND_URL=http://localhost:3000
        echo FRONTEND_URL=http://localhost:3000 >> .env
    )
    
    findstr /C:"EMAIL_HOST" .env >nul
    if errorlevel 1 (
        echo ⚠️  Email configuration not found in .env
        echo Please add email configuration to .env:
        echo   EMAIL_HOST=smtp.gmail.com
        echo   EMAIL_PORT=587
        echo   EMAIL_HOST_USER=your-email@gmail.com
        echo   EMAIL_HOST_PASSWORD=your-app-password
    )
) else (
    echo ⚠️  .env file not found
    echo Please create .env file from .env.example
)
echo.

REM Collect static files (if needed)
echo 📦 Collecting static files...
python manage.py collectstatic --noinput --clear
echo.

echo ✅ Setup complete!
echo.
echo 📋 Next steps:
echo 1. Update .env with FRONTEND_URL and email configuration
echo 2. Create email templates in templates/emails/
echo 3. Run tests: python manage.py test
echo 4. Start server: python manage.py runserver
echo.
echo 📚 Documentation:
echo - Security fixes: SECURITY_FIXES_APPLIED.md
echo - Complete report: COMPLETE_IMPLEMENTATION_REPORT.md
echo.
echo 🎉 All done! Happy coding!
pause
