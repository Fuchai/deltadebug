import subprocess
import pathlib
from pathlib import Path
import os
import shutil


class Delta:
    def __init__(self, test_binary=None, root_directory=None, yesterday_directory=None, today_directory=None):

        self.test_binary = Path(test_binary).resolve()
        self.root_directory = Path(root_directory).resolve()
        self.yesterday_directory = Path(yesterday_directory).resolve()
        self.today_directory = Path(today_directory).resolve()
        self.debug_flag = False

    def pre_run(self):
        ytd_ret = self.test_ytd()
        today_ret = self.test_today()


    def test_ytd(self):
        ytd_run = subprocess.run([str(self.test_binary), str(self.yesterday_directory)])
        if self.debug_flag:
            print("yesterday test return code", ytd_run.returncode)
        return ytd_run.returncode

    def test_today(self):
        today_run = subprocess.run([str(self.test_binary), str(self.today_directory)])
        if self.debug_flag:
            print("yesterday test return code", today_run.returncode)
        return today_run.returncode

    def debug(self, create_patch=True, print_results=True):
        """
        Delta debug.

        :param algo: Pick which algorithm you want to use, 1 or 2.
        :param create_patch: If you want to create a patch for the minimal set of failure inducing changes
        :param print_results: If you want to print out the results of the incremental patches in the minimal set
        of failure inducing changes
        """
        self.create_tmp_folder()
        os.chdir(self.tmp_dir)
        big_patch_path = self.tmp_dir / "bigpatch"
        with big_patch_path.open("w+") as big_patch:
            # create a diff patch
            subprocess.run(["diff", "-u", "-r", str(self.yesterday_directory), str(self.today_directory)], stdout=big_patch)

            # splitpatch into hunks
            subprocess.run(["splitpatch", "-H", str(big_patch_path)], stderr=subprocess.PIPE)

        incrementals = []
        for p in self.tmp_dir.iterdir():
            if p.suffix == ".patch":
                incrementals.append(p)

        if self.debug_flag:
            print(incrementals)
        minimal_patches = self.algo1(incrementals, set())


        if self.debug_flag:
            print(minimal_patches)
        if print_results:
            print("The minimal set of failure-inducing changes includes the following:")
            for p in minimal_patches:
                print(str(p))

        if create_patch:
            combined_path = self.tmp_dir / "combined"
            minimal_patch_path = self.tmp_dir / "minimal_patch"
            shutil.copy(combined_path, minimal_patch_path)
            print("The minimal patch is created at location", minimal_patch_path)

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

    def algo2(self, patches, fixed, n):
        if len(patches) == 1:
            return patches

        set_size = len(patches) // n
        cis = []
        for i in range(n):
            if i == n - 1:
                cis.append(set(patches[i * set_size:]))
            else:
                cis.append(set(patches[i * set_size: (i + 1) * set_size]))

        # case 2
        tis = []
        for ci in cis:
            ti = self.test_patches(ci)
            tis.append(ti)
            if ti == 1:
                return self.algo2(ci, fixed, 2)

        # case 3
        complements = []
        complement_tis = []
        if n == 2:
            complements = [cis[1], cis[0]]
            complement_tis = [tis[1], tis[0]]
            if tis[0] == 0 and tis[1] == 0:
                return self.algo2(cis[0], cis[1] | fixed, 2) | self.algo2(cis[1], cis[0] | fixed, 2)
        else:
            complements = [set(patches) - ci for ci in cis]
            for i, complement in enumerate(complements):
                ci = cis[i]

                complement_ti = self.test_patches(complement)
                complement_tis.append(complement_ti)
                if complement_tis == 0 and tis[i] == 0:
                    return self.algo2(ci, complement | fixed, 2) | self.algo2(complement, ci | fixed, 2)

        # case 4
        for i, complement in enumerate(complements):
            ci = cis[i]
            complement_ti = complement_tis[i]
            ti = tis[i]
            if ti == -1 and complement_ti == 0:
                return self.algo2(ci, complement | fixed, 2)

        # case 5
        # prep
        c_prime = set(patches)
        for complement, complement_ti in zip(complements, complement_tis):
            if complement_ti == 1:
                c_prime = c_prime & complement

        r_prime = fixed
        for ci, ti in zip(cis, tis):
            if ti == 0:
                r_prime = r_prime | ci

        n_prime = min(len(c_prime), 2*n)

        # for real
        if n < len(patches):
            return self.algo2(c_prime, r_prime, n_prime)

        # case 6
        return c_prime


    def test_patches(self, patches):
        # combine patches together
        plist = [str(p) for p in patches]
        combined_path = self.tmp_dir / "combined"
        if combined_path.exists():
            os.remove(combined_path)
        with combined_path.open("w") as combined:
            if len(plist) == 1:
                shutil.copy(plist[0], combined_path)
            else:
                subprocess.run(["combinediff", "-q"] + plist, stdout=combined)

        # patch the files
        with combined_path.open("r") as combined:
            if self.debug_flag:
                subprocess.run(["patch", "-p0", "-d/"], stdin=combined)
            else:
                subprocess.run(["patch", "-p0", "-d/"], stdin=combined, stdout=subprocess.DEVNULL)

            # apply, test ytd, and revert
            ret_code = self.test_ytd()

            # with combined_path.open("r") as combined:
            if self.debug_flag:
                subprocess.run(["patch", "-R", "-p0", "-d/"], stdin=combined)
            else:
                subprocess.run(["patch", "-R", "-p0", "-d/"], stdin=combined, stdout=subprocess.DEVNULL)

        return ret_code

    def create_tmp_folder(self):
        this_path = pathlib.Path(__file__).parent.absolute()
        self.tmp_dir = this_path / "tmp"
        if self.tmp_dir.exists() and self.tmp_dir.is_dir():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir()


def quick_main():
    tb = "test_dummy.py"
    yd = "patches/expcp"
    td = "patches/expcp2"
    root = "patches"
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
