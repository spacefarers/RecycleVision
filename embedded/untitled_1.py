# Untitled - By: spacefarers - Sun Nov 2 2025

import sys
import os

for i in range(0, 2):
    print("hello canmv")
    print("hello ", end="canmv\n")

print("implementation:", sys.implementation)
print("platform:", sys.platform)
print("path:", sys.path)
print("Python version:", sys.version)
print(os.listdir("/sdcard"))
