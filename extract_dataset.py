#!/usr/bin/env python3
"""
RAR Extraction Script for GBM Dataset
Handles RAR extraction with multiple fallback methods
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_package(package):
    """Install a Python package using pip"""
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def try_rarfile_extraction(rar_path, extract_to):
    """Try extraction using rarfile package"""
    try:
        import rarfile
        print("Using rarfile library...")
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(extract_to)
        return True
    except ImportError:
        print("rarfile not found, attempting to install...")
        if install_package("rarfile"):
            try:
                import rarfile
                print("Using newly installed rarfile...")
                with rarfile.RarFile(rar_path) as rf:
                    rf.extractall(extract_to)
                return True
            except Exception as e:
                print(f"rarfile extraction failed after installation: {e}")
                return False
        return False
    except Exception as e:
        print(f"rarfile extraction failed: {e}")
        return False

def try_patoolib_extraction(rar_path, extract_to):
    """Try extraction using patoolib package"""
    try:
        import patoolib
        print("Using patoolib library...")
        patoolib.extract_archive(rar_path, outdir=extract_to)
        return True
    except ImportError:
        print("patoolib not found, attempting to install...")
        if install_package("patoolib"):
            try:
                import patoolib
                print("Using newly installed patoolib...")
                patoolib.extract_archive(rar_path, outdir=extract_to)
                return True
            except Exception as e:
                print(f"patoolib extraction failed after installation: {e}")
                return False
        return False
    except Exception as e:
        print(f"patoolib extraction failed: {e}")
        return False

def try_7zip_extraction(rar_path, extract_to):
    """Try extraction using 7-Zip"""
    seven_zip_paths = [
        "C:\\Program Files\\7-Zip\\7z.exe",
        "C:\\Program Files (x86)\\7-Zip\\7z.exe",
        "7z"  # If in PATH
    ]
    
    for seven_zip in seven_zip_paths:
        try:
            cmd = [seven_zip, "x", rar_path, f"-o{extract_to}", "-y"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Extracted using 7-Zip")
                return True
        except FileNotFoundError:
            continue
    return False

def try_winrar_extraction(rar_path, extract_to):
    """Try extraction using WinRAR"""
    winrar_paths = [
        "C:\\Program Files\\WinRAR\\WinRAR.exe",
        "C:\\Program Files (x86)\\WinRAR\\WinRAR.exe",
        "winrar"  # If in PATH
    ]
    
    for winrar in winrar_paths:
        try:
            cmd = [winrar, "x", "-y", rar_path, extract_to]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Extracted using WinRAR")
                return True
        except FileNotFoundError:
            continue
    return False

def try_system_unrar(rar_path, extract_to):
    """Try extraction using system unrar command"""
    try:
        result = subprocess.run(["unrar", "x", rar_path, extract_to], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Extracted using unrar")
            return True
        return False
    except FileNotFoundError:
        return False

def verify_extraction(extract_dir):
    """Verify that extraction worked by checking for key files"""
    key_files = ["train_labels.csv", "train", "test"]
    
    print(f"Verifying extraction in: {extract_dir}")
    
    # Check if the directory exists
    if not extract_dir.exists():
        print(f"✗ Directory {extract_dir} does not exist")
        return False
    
    # List contents for debugging
    contents = list(extract_dir.iterdir())
    print(f"Found {len(contents)} items in extraction directory:")
    for item in contents[:10]:  # Show first 10 items
        print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
    
    # Check for key files
    missing_files = []
    for key_file in key_files:
        file_path = extract_dir / key_file
        if not file_path.exists():
            missing_files.append(key_file)
        else:
            print(f"✓ Found {key_file}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    return True

def clean_incomplete_extraction(extract_dir):
    """Remove incomplete extraction directory"""
    try:
        if extract_dir.exists():
            print(f"Cleaning up incomplete extraction: {extract_dir}")
            shutil.rmtree(extract_dir)
    except Exception as e:
        print(f"Warning: Could not clean up {extract_dir}: {e}")

def main():
    rar_path = Path("data/rsna-miccai-brain-tumor-radiogenomic-classification.rar")
    extract_to = Path("data")
    extract_dir = extract_to / "rsna-miccai-brain-tumor-radiogenomic-classification"
    
    print("="*60)
    print("RAR Dataset Extraction Tool")
    print("="*60)
    
    if not rar_path.exists():
        print(f"❌ Error: RAR file not found at {rar_path}")
        print("Please download the dataset first.")
        sys.exit(1)
    
    print(f"📁 RAR file: {rar_path}")
    print(f"📂 Extract to: {extract_to}")
    print(f"🎯 Target directory: {extract_dir}")
    
    # Check if already extracted
    if extract_dir.exists() and verify_extraction(extract_dir):
        print("✅ Dataset already extracted and verified!")
        sys.exit(0)
    
    # Try different extraction methods
    methods = [
        ("7-Zip", try_7zip_extraction),
        ("WinRAR", try_winrar_extraction), 
        ("rarfile (Python)", try_rarfile_extraction),
        ("patoolib (Python)", try_patoolib_extraction),
        ("system unrar", try_system_unrar)
    ]
    
    print("\n🔄 Trying extraction methods...")
    
    for method_name, method_func in methods:
        print(f"\n📦 Trying {method_name}...")
        
        try:
            if method_func(str(rar_path), str(extract_to)):
                print(f"✓ {method_name} completed extraction")
                
                if verify_extraction(extract_dir):
                    file_count = sum(1 for _ in extract_dir.rglob("*") if _.is_file())
                    print(f"✅ SUCCESS! Extracted {file_count} files using {method_name}")
                    print("="*60)
                    sys.exit(0)
                else:
                    print(f"✗ {method_name} extraction incomplete - cleaning up")
                    clean_incomplete_extraction(extract_dir)
            else:
                print(f"✗ {method_name} failed to extract")
        except Exception as e:
            print(f"✗ {method_name} error: {e}")
            clean_incomplete_extraction(extract_dir)
    
    # All methods failed
    print("\n" + "="*60)
    print("❌ ALL EXTRACTION METHODS FAILED!")
    print("="*60)
    print("\nThe RAR file requires external tools to extract.")
    print("\n🔧 SOLUTIONS:")
    print("1. Install 7-Zip: https://www.7-zip.org/")
    print("   - Download and install")
    print("   - Run this script again")
    print("\n2. Install WinRAR: https://www.winrar.es/")
    print("   - Download and install")
    print("   - Run this script again")
    print("\n3. Manual extraction:")
    print(f"   - Right-click: {rar_path}")
    print("   - Select 'Extract Here' or 'Extract to...'")
    print(f"   - Ensure files are extracted to: {extract_dir}")
    print("\n4. Use online RAR extractor (if file size allows)")
    print("\n⚠️  After extraction, verify these files exist:")
    print(f"   - {extract_dir}/train_labels.csv")
    print(f"   - {extract_dir}/train/ (directory)")
    print(f"   - {extract_dir}/test/ (directory)")
    print("\n🔄 Then run this script again to verify extraction.")
    print("="*60)
    sys.exit(1)

if __name__ == "__main__":
    main()
