#!/usr/bin/env python3
"""
Script to create all necessary __init__.py files
Run this from the project root directory
"""

import os
from pathlib import Path

# Define all directories that need __init__.py
directories = [
    'src',
    'src/data',
    'src/models',
    'src/training',
    'src/evaluation',
    'src/utils',
]

# Content for each __init__.py file
init_contents = {
    'src': '"""Source code for NLP sentiment analysis project."""\n',
    'src/data': '"""Data loading and preprocessing modules."""\n\nfrom .dataset import DataPreprocessor, MultiDomainSentimentDataset\n\n__all__ = [\'DataPreprocessor\', \'MultiDomainSentimentDataset\']\n',
    'src/models': '"""Model definitions."""\n',
    'src/training': '"""Training module for sentiment analysis."""\n\nfrom .trainer import SentimentTrainer\n\n__all__ = [\'SentimentTrainer\']\n',
    'src/evaluation': '"""Evaluation and metrics modules."""\n',
    'src/utils': '"""Utility functions."""\n',
}

def create_init_files():
    """Create __init__.py files in all required directories."""
    project_root = Path.cwd()
    
    print("Creating __init__.py files...")
    print("="*60)
    
    for directory in directories:
        dir_path = project_root / directory
        
        # Create directory if it doesn't exist
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py file
        init_file = dir_path / '__init__.py'
        
        # Get content for this directory
        content = init_contents.get(directory, '')
        
        # Write the file
        with open(init_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Created: {init_file}")
    
    print("="*60)
    print("✅ All __init__.py files created successfully!")
    print("\nYou can now run your training script.")

if __name__ == '__main__':
    create_init_files()