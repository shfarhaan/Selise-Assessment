"""
Setup and Installation Script for Agentic RAG System
Run this to verify your environment and install dependencies
"""
import sys
import subprocess
import os
from pathlib import Path

def check_virtual_environment():
    """Check if running in a virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print(f"✅ Running in virtual environment")
        print(f"   Location: {sys.prefix}")
        return True
    else:
        print("⚠️  Not running in a virtual environment")
        print("   Recommendation: Use a virtual environment to isolate dependencies")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, you have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_file(filepath):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ Missing: {filepath}")
        return False

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        return False

def check_env_file():
    """Check for .env file"""
    if os.path.exists(".env"):
        print("✅ .env file found")
        # Check if API key and endpoint are set
        with open(".env", "r") as f:
            content = f.read()
            has_api_key = "GEMINI_API_KEY" in content and "your-" not in content
            
            if has_api_key:
                print("✅ Gemini API key appears to be configured")
                return True
            else:
                if not has_api_key:
                    print("⚠️  GEMINI_API_KEY not properly set in .env")
                return False
    else:
        print("❌ .env file not found")
        print("   Create .env file with GEMINI_API_KEY")
        return False

def check_project_structure():
    """Verify project structure"""
    print("\n📁 Checking project structure...")
    
    required_files = [
        "app.py",
        "cli.py",
        "requirements.txt",
        "src/config.py",
        "src/document_processor.py",
        "src/embeddings.py",
        "src/retriever.py",
        "src/agent.py",
    ]
    
    all_exist = True
    for file in required_files:
        if not check_file(file):
            all_exist = False
    
    return all_exist

def verify_imports():
    """Try importing key packages"""
    print("\n🔍 Verifying package imports...")
    
    packages = {
        "streamlit": "streamlit",
        "google.genai": "google-genai",
        "faiss": "faiss-cpu",
        "numpy": "numpy",
        "dotenv": "python-dotenv"
    }
    
    all_ok = True
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} not installed")
            all_ok = False
    
    return all_ok

def main():
    """Main setup routine"""
    print("="*60)
    print("Agentic RAG System - Setup Verification")
    print("="*60)
    
    # Check Python version
    print("\n🐍 Checking Python version...")
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        return False
    
    # Check virtual environment
    print("\n🔧 Checking virtual environment...")
    in_venv = check_virtual_environment()
    if not in_venv:
        print("\n💡 How to create a virtual environment:")
        print("   1. Run: install.bat (automatic)")
        print("   2. Or manually:")
        print("      python -m venv venv")
        print("      venv\\Scripts\\activate  (Windows)")
        print("      source venv/bin/activate  (Mac/Linux)")
        print("\n   Continue anyway? (y/n): ", end="")
        response = input().lower()
        if response != 'y':
            print("\n⚠️  Exiting. Please set up virtual environment first.")
            return False
    
    # Check project structure
    if not check_project_structure():
        print("\n❌ Setup failed: Missing required files")
        return False
    
    # Check if packages are installed
    print("\n📦 Checking installed packages...")
    packages_installed = verify_imports()
    
    if not packages_installed:
        print("\n⚠️  Some packages missing.")
        response = input("\nInstall dependencies now? (y/n): ").lower()
        if response == 'y':
            if not install_dependencies():
                print("\n❌ Setup failed: Could not install dependencies")
                return False
            # Verify again
            if not verify_imports():
                print("\n❌ Setup failed: Packages still missing after installation")
                return False
        else:
            print("\n⚠️  Run: pip install -r requirements.txt")
            return False
    
    # Check .env file
    print("\n🔑 Checking API key configuration...")
    env_ok = check_env_file()
    
    # Final summary
    print("\n" + "="*60)
    if env_ok:
        print("✅ Setup Complete! You're ready to go!")
        print("\nNext steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Or: python cli.py info")
    else:
        print("⚠️  Almost there!")
        print("\nNext steps:")
        print("  1. Create .env file with your Gemini credentials")
        print("  2. Set GEMINI_API_KEY")
        print("  3. Run: streamlit run app.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
