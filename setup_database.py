#!/usr/bin/env python3
"""
Database setup script for TrendTracker
This script helps you set up the PostgreSQL database for the stock prediction application.
"""

import os
import sys
from dotenv import load_dotenv

def create_env_file():
    """Create .env file with database configuration"""
    env_content = """# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/trendtracker

# API Configuration
API_URL=https://trendtracker-uvqu.onrender.com

# Application Settings
DEBUG=True
ENVIRONMENT=development
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
        print("⚠️  Please update the DATABASE_URL with your actual PostgreSQL credentials")
    else:
        print("ℹ️  .env file already exists")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        ('psycopg2-binary', 'psycopg2'),
        ('sqlalchemy', 'sqlalchemy'),
        ('alembic', 'alembic'),
        ('python-dotenv', 'dotenv')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def initialize_database():
    """Initialize the database tables"""
    try:
        from database import init_db
        init_db()
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        print("Make sure PostgreSQL is running and DATABASE_URL is correct")
        return False

def run_migrations():
    """Run database migrations"""
    try:
        import subprocess
        result = subprocess.run(['alembic', 'upgrade', 'head'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Database migrations completed successfully")
            return True
        else:
            print(f"❌ Migration failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Alembic not found. Please install it: pip install alembic")
        return False
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 TrendTracker Database Setup")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Step 1: Create .env file
    print("\n1. Creating environment file...")
    create_env_file()
    
    # Step 2: Check dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Step 3: Initialize database
    print("\n3. Initializing database...")
    if not initialize_database():
        print("\n💡 Troubleshooting tips:")
        print("- Make sure PostgreSQL is installed and running")
        print("- Create a database named 'trendtracker'")
        print("- Update DATABASE_URL in .env file with correct credentials")
        print("- Example: DATABASE_URL=postgresql://username:password@localhost:5432/trendtracker")
        sys.exit(1)
    
    # Step 4: Run migrations
    print("\n4. Running database migrations...")
    if not run_migrations():
        print("⚠️  Migrations failed, but database tables were created")
    
    print("\n🎉 Database setup completed!")
    print("\nNext steps:")
    print("1. Update DATABASE_URL in .env file with your PostgreSQL credentials")
    print("2. Run: uvicorn forecast:app --reload")
    print("3. Run: streamlit run app.py")
    print("4. Open http://localhost:8501 in your browser")

if __name__ == "__main__":
    main() 