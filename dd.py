import subprocess
import pathlib
import os
import shutil


class Delta:
    def __init__(self, test_binary=None, root_directory=None, yesterday_directory=None, today_directory=None):
        self.test_binary = test_binary
        self.root_directory = root_directory
        self.yesterday_directory = yesterday_directory
        self.today_directory = today_directory
        self.debug_flag = True

    def pre_run(self):
        """
        run the test binary on yesterday and today to check the properties

        :return:
        """
        ytd_ret = self.test_ytd()
        today_ret = self.test_today()



        assert ytd_ret == 0, "Yesterday's code did not work?"
        assert today_ret == 1, "Today's code did not break?"

    def test_ytd(self):
        ytd_run = subprocess.run([self.test_binary, self.yesterday_directory])
        if self.debug_flag:
            print("yesterday test return code", ytd_run.returncode)
        return ytd_run.returncode

    def test_today(self):
        today_run = subprocess.run([self.test_binary, self.today_directory])
        if self.debug_flag:
            print("yesterday test return code", today_run.returncode)
        return today_run.returncode

    def debug(self, create_patch=True, print_results=True):
        """
        Runs algorithm one on the paper

        :return:
        """
        self.create_tmp_folder()
        os.chdir(self.tmp_dir)
        big_patch_path = self.tmp_dir / "bigpatch"
        with big_patch_path.open("w+") as big_patch:
            # create a diff patch
            subprocess.run(["diff", "-u", "-r", "--new-file", self.yesterday_directory, self.today_directory], stdout=big_patch)

            # splitpatch into hunks
            subprocess.run(["splitpatch", "-H", str(big_patch_path)], stderr=subprocess.PIPE)

        incrementals = []
        for p in self.tmp_dir.iterdir():
            if p.suffix == ".patch":
                incrementals.append(p)

        self.algo1(incrementals, set())

    def algo1(self, patches, fixed):
        if len(patches) == 1:
            return patches

        # split the patches into two sets
        c1 = set(patches[0:len(patches) // 2])
        c2 = set(patches[len(patches) // 2:])

        if self.test_patches(c1 | fixed) == 1:
            return self.algo1(c1, fixed)
        elif self.test_patches(c2 | fixed) == 1:
            return self.algo1(c2, fixed)
        else:
            return self.algo1(c1, c2 | fixed) | self.algo1(c2, c1 | fixed)

    def test_patches(self, patches):
        # combine patches together
        plist = [str(p) for p in patches]
        combined_path = self.tmp_dir / "combined"
        if combined_path.exists():
            os.remove(combined_path)
        with combined_path.open("w") as combined:
            if len(plist)==1:
                shutil.copy(plist[0],combined_path)
            else:
                subprocess.run(["combinediff", "-q"] + plist, stdout=combined)

        # patch the files
        with combined_path.open("r") as combined:
            subprocess.run(["patch", "-p0", "-d/"], stdin=combined, stderr=subprocess.PIPE)

            # apply, test ytd, and revert
            ret_code = self.test_ytd()
            subprocess.run(["patch", "-R", "-p0", "-d/"], stdin=combined, stderr=subprocess.STDOUT)

        return ret_code

    def create_tmp_folder(self):
        this_path = pathlib.Path(__file__).parent.absolute()
        self.tmp_dir = this_path / "tmp"
        if self.tmp_dir.exists() and self.tmp_dir.is_dir():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir()


def quick_main():
    tb = "/home/jasonhu/Desktop/pydd/test_dummy.py"
    yd = "/home/jasonhu/Desktop/pydd/patches/expcp"
    td = "/home/jasonhu/Desktop/pydd/patches/expcp2"
    root = "/home/jasonhu/Desktop/pydd/patches"

    delta = Delta(tb, root, yd, td)
    delta.pre_run()
    delta.debug()


def main():
    tb = input("Please enter the test binary directory\n")
    root = input("Please enter the root directory where your yesterday and today directories reside\n")
    yd = input("Please enter the yesterday directory where your program works\n")
    td = input("Please enter the today directory where your program does not work :(\n")

    delta = Delta(tb, root, yd, td)
    delta.pre_run()
    delta.debug()


if __name__ == '__main__':
    quick_main()
