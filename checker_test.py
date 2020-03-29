#!/usr/bin/python3

import sys
import os
import subprocess
from pathlib import Path

def main():
    dd_path=Path(sys.argv[1])/"test_checker.py"
    pro=subprocess.run(["python3", str(dd_path)],stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pro.returncode


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)