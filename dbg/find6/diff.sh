#!/bin/bash
cd /root/Desktop/find6/find/
#git diff HEAD > ../find6.patch
#diff -x "*~" -Naur  /root/Desktop/find6/find > ../find6.patch
git diff find6 > ../find6.patch
echo ""
echo "Please copy the contents of find6.patch into"
echo "the appropriate field at http://bit.do/find-091557f6."
echo ""
