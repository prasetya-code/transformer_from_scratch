from pathlib import Path
import sys

# 
file_path = Path(__file__).resolve().parent
project_root = file_path.parent

# Jangan lupa diubah ke string untuk sys.path
sys.path.append(str(project_root)) 