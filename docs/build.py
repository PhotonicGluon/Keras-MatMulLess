"""
Builds the documentation in the source folder.
"""

import os
import sys

# Set current working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run sphinx auto-builder
exit_code = os.system("sphinx-autobuild source build")
if exit_code != 0:
    sys.exit(exit_code)
