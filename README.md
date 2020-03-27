```
diff -u -r exp expcp > mp2
splitpatch mp2
combinediff -q a.patch b.patch > trythis2
patch -p0 < trythis2

diff -u -r ../expcp ../expcp2 | fgrep -v '^Only in'  
```


## Dependencies
```splitpatch``` command from **splitpatch**
https://github.com/benjsc/splitpatch

```combinediff``` command from **patchutils**
https://directory.fsf.org/wiki/Patchutils

```patch``` and ```diff``` commands

## Specifications

We need two directories containing the source code before and after the change, placed in a source root directory.
All of these files will be modified, so backup for yourself.

We need a test binary that takes only one argument that is a source code directory and returns 0, 1 or -1. 
The return value 0 indicates that the test succeeds,
1 indicates that the test produces the failure it's intended to capture, 
and -1 indicates an indeterminate result.

This pydd program will return the minimal failure-inducing set of patches, assuming
monotony, unambiguity, and consistency. We provide API tools to create a combined patch of the
minimal failure-inducing set of changes. The API tool can also print out the patches through command line.


## Reference
Zeller, Andreas. "Yesterday, my program worked. Today, it does not. Why?." 
ACM SIGSOFT Software engineering notes 24.6 (1999): 253-267.