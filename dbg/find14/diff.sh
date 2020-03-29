#!/bin/bash
cd /root/Desktop/find14/find/
#git diff HEAD > ../find14.patch
#diff -x "*~" -Naur  /root/Desktop/find14/find > ../find14.patch
git diff find14 > ../find14.patch
echo ""
echo "Please copy the contents of find14.patch into"
echo "the appropriate field at http://bit.do/find-ff248a20."
echo ""
