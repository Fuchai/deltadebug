#!/usr/bin/python3

import sys
from os import listdir


def main():
    source_dir = sys.argv[1]

    for f in listdir(source_dir):
        if (f == "bug"):
            return 1

    return 0


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
