#!/usr/bin/env python
"""
IntelliSafe Dependency Installer
Run this script to install all required Python dependencies
"""

import subprocess
import sys
import argparse
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 to avoid UnicodeEncodeErrors on Windows terminals with emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Load Torch before Qt to avoid a Windows DLL initialization conflict where
# importing PyQt6 first can prevent torch\lib\c10.dll from initializing.
try:
    import torch  # noqa: F401
except ImportError:
    pass

GLINER_MODEL_CHOICES = [
    "urchade/gliner_small-v2.1",
    "urchade/gliner_base-v2.1",
    "urchade/gliner_medium-v2.1",
    "urchade/gliner_large-v2.1",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Install IntelliSafe dependencies")
    parser.add_argument(
        "--gliner-model",
        default=GLINER_MODEL_CHOICES[1],
        choices=GLINER_MODEL_CHOICES,
        help="GLiNER model to prefetch after installation",
    )
    parser.add_argument(
        "--skip-gliner-download",
        action="store_true",
        help="Install packages only and skip GLiNER model prefetch",
    )
    return parser.parse_args()


def prefetch_gliner_model(project_path: Path, model_name: str) -> bool:
    """Attempt to download/cache the selected GLiNER model."""
    print("\n" + "=" * 70)
    print(f"🧠 Prefetching GLiNER model: {model_name}")
    print("=" * 70)

    probe = f"""
from gliner import GLiNER
if hasattr(GLiNER, 'from_pretrained'):
    model = GLiNER.from_pretrained({model_name!r})
else:
    model = GLiNER({model_name!r})
print('Loaded:', {model_name!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(project_path),
        check=False,
    )

    if result.returncode == 0:
        print(f"✅ GLiNER model cached: {model_name}")
        return True

    print(f"⚠️  Could not prefetch GLiNER model: {model_name}")
    print("   The model will be downloaded on first use if the package supports it.")
    return False


def install_dependencies():
    """Install all dependencies from requirements.txt"""
    
    print("=" * 70)
    print("📦 IntelliSafe - Python Dependencies Installer")
    print("=" * 70)
    
    # Get project path
    project_path = Path(__file__).parent
    requirements_file = project_path / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ Error: requirements.txt not found at {requirements_file}")
        return False
    
    print(f"\n📂 Project Path: {project_path}")
    print(f"📋 Requirements File: {requirements_file}")
    print(f"🐍 Python Executable: {sys.executable}")
    print(f"📌 Python Version: {sys.version}")
    
    print("\n" + "=" * 70)
    print("🚀 Starting dependency installation...")
    print("=" * 70)
    
    args = parse_args()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "--upgrade"],
            cwd=str(project_path),
            check=False
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✅ SUCCESS! All dependencies installed successfully!")
            print("=" * 70)
            print("\n📝 Next Steps:")
            print("   1. Run the application: python main.py")
            print("   2. Read QUICKSTART.md for getting started")
            print("   3. Test file upload functionality")
            if not args.skip_gliner_download:
                prefetch_gliner_model(project_path, args.gliner_model)
            return True
        else:
            print("\n" + "=" * 70)
            print(f"⚠️  Installation completed with warnings (return code: {result.returncode})")
            print("=" * 70)
            print("\n📝 You may need to install some packages manually:")
            print("   pip install PyQt6")
            print("   pip install gliner presidio-analyzer")
            if not args.skip_gliner_download:
                prefetch_gliner_model(project_path, args.gliner_model)
            return False
            
    except Exception as e:
        print(f"\n❌ Error during installation: {e}")
        return False

def check_critical_packages():
    """Check if critical packages are installed"""
    print("\n" + "=" * 70)
    print("🔍 Checking critical packages...")
    print("=" * 70)
    
    critical_packages = {
        'PyQt6': 'UI Framework',
        'gliner': 'Graph-based NER (GLiNER)',
        'presidio_analyzer': 'Presidio Analyzer',
        'transformers': 'ML Models',
        'fitz': 'PDF Processing (PyMuPDF)',
        'docx': 'Word Documents (python-docx)',
        'cv2': 'Image Processing (opencv-python)',
    }
    
    all_ok = True
    for package, description in critical_packages.items():
        try:
            __import__(package)
            print(f"✅ {package:20} - {description}")
        except ImportError:
            print(f"❌ {package:20} - {description} (NOT INSTALLED)")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    # Install dependencies
    success = install_dependencies()
    
    # Check critical packages
    check_critical_packages()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 Ready to use IntelliSafe!")
        print("=" * 70)
        print("\n   Run: python main.py")
    else:
        print("\n" + "=" * 70)
        print("⚠️  Please resolve the above issues and try again")
        print("=" * 70)
    
    sys.exit(0 if success else 1)
