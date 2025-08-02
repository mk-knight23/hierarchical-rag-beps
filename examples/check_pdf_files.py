#!/usr/bin/env python3
"""
Utility script to check what PDF files are available in data/raw
"""

import sys
from pathlib import Path

def check_pdf_files():
    """Check for PDF files in data/raw directory"""
    
    data_raw = Path("data/raw")
    
    if not data_raw.exists():
        print(f"❌ Directory {data_raw} does not exist")
        return
    
    # List all files
    all_files = list(data_raw.glob("*"))
    pdf_files = list(data_raw.glob("*.pdf"))
    txt_files = list(data_raw.glob("*.txt"))
    
    print(f"\n📁 Directory: {data_raw.absolute()}")
    print(f"📊 Total files: {len(all_files)}")
    print(f"📄 PDF files: {len(pdf_files)}")
    print(f"📝 Text files: {len(txt_files)}")
    
    if pdf_files:
        print("\n📄 PDF Files:")
        for pdf in sorted(pdf_files):
            size_mb = pdf.stat().st_size / 1024 / 1024
            print(f"   • {pdf.name} ({size_mb:.1f} MB)")
    
    if txt_files:
        print("\n📝 Text Files:")
        for txt in sorted(txt_files):
            size_kb = txt.stat().st_size / 1024
            print(f"   • {txt.name} ({size_kb:.0f} KB)")
    
    if not pdf_files and not txt_files:
        print("\n⚠️  No PDF or text files found in data/raw")
        print("💡 You may need to add your OECD BEPS PDF documents to this directory")

if __name__ == "__main__":
    check_pdf_files()