Find is a command-line utility that searches through one or more directory trees
of a file system, locates files based on some user-specified criteria and applies
a user-specified action on each matched file. The possible search criteria include a
pattern to match against the file name or a time range to match against the modification
time or access time of the file. By default, find returns a list of all files below
the current working directory.


BUG DIAGNOSIS

1) Visit http://bit.do/find-ff248a20?entry.594570799=MjU4OTJhOW to read the simplified
bug report specific to this error find14.

NOTE THAT EVERY ERROR HAS ITS OWN LINK!

2) Run the following test case. If the exit code is 0, the test case has passed,
otherwise it failed. You can check the exit code with "echo $?".

/root/Desktop/find14/test/test.sh /root/Desktop/find14/find



3) Note down the DIAGNOSIS STARTING TIME for this error.

4) You may want to start bug diagnosis by understanding the runtime actions leading
to the error that is reproduced by the test case. Of course, you can use any tools
that are available on this machine or on the internet. Technically, all bugs can be
explained in *.c or *.h files.

5) Note down the DIAGNOSIS ENDING TIME for this error.

6) Answer all questions on the first page of the questionnaire at
http://bit.do/find-ff248a20?entry.594570799=MjU4OTJhOW and click 'Next'.


BUG FIXING

7) Note down the FIXING STARTING TIME for this error.

8) Fix the error. Again, you can use any tool that is available to you.

9) Note down the FIXING ENDING TIME for this error.

10) Answer all questions on the second page of the questionnaire at
http://bit.do/find-ff248a20?entry.594570799=MjU4OTJhOW .

--> Execute /root/Desktop/find14/find/../diff.sh and copy the contents of
/root/Desktop/find14/find/../find14.patch to the field provided in the questionnaire.

--> Execute /root/Desktop/find14/find/../restore.sh to undo all changes if you have
any troubles and want to start anew.

--> Execute make in the folder /root/Desktop/find14/find to compile the program after
it was changed.

When you are finished, click 'Submit'

You can now continue with the next error.

In case you need to use sudo privileges on the VM:
* user     = root
* password = corebench

Your uuid is: MjU4OTJhOW

If you have any questions about the infrastructure or the questionnaire, do not hesitate
to contact us immediately!
