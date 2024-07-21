"""
Cleans up the source folder, removing any generated files.
"""

import glob
import os
import shutil
import sys

# Set current working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Main clean
exit_code = os.system("make clean")
if exit_code != 0:
    sys.exit(exit_code)

# Remove generated folders
generated_folders = glob.glob("source/**/generated", recursive=True)

for generated_folder in generated_folders:
    shutil.rmtree(generated_folder)
    print(f"Removed '{generated_folder}'")

shutil.rmtree("build")
print(f"Removed 'build'")

print("Done cleaning.")
