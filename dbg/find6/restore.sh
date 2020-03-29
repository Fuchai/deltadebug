#!/bin/bash
choice=0
read -p "All your changes will be lost. Are you sure you want to rollback and restore the original (buggy) program (y/n)?" choice
case "$choice" in 
  y|Y ) echo "Okay";;
  n|N ) exit 0 ;;
  * ) echo "Did not understand. Try again."; exit 1 ;;
esac

cd /root/Desktop/find6/find
git reset --hard
"~/find.compile.sh" "/root/Desktop/find6/find" b46b0d89 "/root/corebench" 2>&1 >& /dev/null
