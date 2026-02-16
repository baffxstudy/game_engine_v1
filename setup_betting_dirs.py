"""
Setup script to create required directories for betting automation framework.
Run this once to create necessary directories.
"""

import os
from pathlib import Path

# Create directories
dirs = [
    "screenshots",
    "logs"
]

for dir_name in dirs:
    dir_path = Path(__file__).parent / dir_name
    dir_path.mkdir(exist_ok=True)
    
    # Create .gitkeep file
    gitkeep = dir_path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
    
    print(f"✓ Created directory: {dir_path}")

print("\n✅ Directory setup complete!")
