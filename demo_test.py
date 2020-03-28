#!/usr/bin/python3

import sys
from os import listdir
import subprocess
from pathlib import Path

# this test runs the delta debugging program
# my delta debugging program should revert all patches applied so that the original files has the same content
# as before
# this test tool checks if that is true

def main():
    # run the delta debugging program


    # check if the files remain the same
    source_dir = Path(sys.argv[1])/"patches"
    referencce="demodir/reference"
    result=subprocess.run(["diff", "-r", referencce, str(source_dir)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    if result.stdout=="" and result.stderr=="":
        return 0
    else:
        return 1


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
