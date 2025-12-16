# setup_project.py
import os
import subprocess
import sys

def setup_project():
    """Setup project Django dengan model IndoBERT"""
    
    print("\n" + "="*70)
    print("HOAX DETECTOR PROJECT SETUP")
    print("="*70)
    
    # 1. Check dependencies
    print("\n[1] Checking dependencies...")
    
    required_packages = [
        'django',
        'torch',
        'transformers',
        'pandas',
        'scikit-learn',
        'numpy',
        'requests',
        'beautifulsoup4',
        'fake-useragent',
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} not found")
    
    # 2. Setup Django database
    print("\n[2] Setting up Django database...")
    try:
        subprocess.run([sys.executable, "manage.py", "makemigrations"], check=True)
        subprocess.run([sys.executable, "manage.py", "migrate"], check=True)
        print("✓ Database setup completed")
    except Exception as e:
        print(f"✗ Error setting up database: {e}")
    
    # 3. Create trained_models directory
    print("\n[3] Creating model directories...")
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    model_path = os.path.join(model_dir, 'indobert_model')
    tokenizer_path = os.path.join(model_dir, 'tokenizer')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tokenizer_path, exist_ok=True)
    
    print(f"✓ Model directory: {model_dir}")
    print(f"✓ Model path: {model_path}")
    print(f"✓ Tokenizer path: {tokenizer_path}")
    
    # 4. Check if model exists
    print("\n[4] Checking model files...")
    
    model_files = os.listdir(model_path) if os.path.exists(model_path) else []
    tokenizer_files = os.listdir(tokenizer_path) if os.path.exists(tokenizer_path) else []
    
    if len(model_files) > 0 and len(tokenizer_files) > 0:
        print("✓ Model files found")
        print(f"  Model files: {len(model_files)} files")
        print(f"  Tokenizer files: {len(tokenizer_files)} files")
    else:
        print("✗ Model files not found")
        print("  Run one of the following:")
        print("  1. python train_model.py (train from scratch)")
        print("  2. python download_indobert.py (download pre-trained)")
    
    # 5. Create superuser
    print("\n[5] Creating superuser...")
    create_superuser = input("Create superuser? (y/n): ").lower().strip()
    
    if create_superuser == 'y':
        try:
            subprocess.run([sys.executable, "manage.py", "createsuperuser"], check=True)
            print("✓ Superuser created")
        except Exception as e:
            print(f"✗ Error creating superuser: {e}")
    
    # 6. Collect static files
    print("\n[6] Collecting static files...")
    try:
        subprocess.run([sys.executable, "manage.py", "collectstatic", "--noinput"], check=True)
        print("✓ Static files collected")
    except Exception as e:
        print(f"✗ Error collecting static files: {e}")
    
    print("\n" + "="*70)
    print("SETUP COMPLETED!")
    print("="*70)
    print("\nTo run the application:")
    print("1. python manage.py runserver")
    print("2. Open http://localhost:8000 in your browser")
    print("\nTo train the model:")
    print("python train_model.py")
    
    return True

if __name__ == "__main__":
    setup_project()