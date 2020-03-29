#!/usr/bin/python3

import sys
import os
import subprocess
from pathlib import Path

def main():
    # for some reason the testing file only works with absolute paths, well.
    

    find_path=Path(sys.argv[1])/"find"
    pro=subprocess.run(["make", str(find_path)])
    tt_path=Path("dbg")/"find6"/"test"/"test.sh"
    tt_path=tt_path.resolve()
    find_path=find_path.resolve()
    pro=subprocess.run([str(tt_path), str(find_path)])
    print("dbg return code " + str(pro.returncode))
    return pro.returncode


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
