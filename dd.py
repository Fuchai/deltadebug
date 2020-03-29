import subprocess
import pathlib
from pathlib import Path
import os
import shutil


class Delta:
    def __init__(self, test_binary=None, root_directory=None, yesterday_directory=None, today_directory=None):
        """

        :param test_binary: the binary that tests the repository. The binary will accept the path as the only argument
        and returns 0 for correct, 1 for buggy and -1 for indeterminate
        :param root_directory: the directory that contains both today's and yesterday's code
        :param yesterday_directory: yesterday it works
        :param today_directory: today it does not work
        """
        os.chdir(Path(__file__).parent)
        self.test_binary = Path(test_binary).resolve()
        self.original_root_directory = Path(root_directory).resolve()
        self.yesterday_directory = Path(yesterday_directory).resolve()
        self.today_directory = Path(today_directory).resolve()
        self.debug_flag = True
        self.patch_directory = None
        self.directory_structure_check()
        self.create_tmp_folder()
        self.copy_root = Path(self.tmp_dir) / self.original_root_directory.name
        self.copy_to_tmp(force=True)
        self.permanently_reroute_to_tmp()
        os.chdir(self.tmp_dir)
        self.big_patch_path = None

    def directory_structure_check(self):
        assert self.yesterday_directory.parent == self.original_root_directory
        assert self.today_directory.parent == self.original_root_directory

    def copy_to_tmp(self, force=False):
        if not self.copy_root.exists():
            shutil.copytree(self.original_root_directory, self.copy_root)
        elif force:
            shutil.rmtree(self.copy_root)
            shutil.copytree(self.original_root_directory, self.copy_root)
        else:
            raise FileExistsError("The tmp folder exists. Use force=True")

    def permanently_reroute_to_tmp(self):
        """
        Paths relative to the tmp folder
        :return:
        """
        self.yesterday_directory = Path(self.original_root_directory.name) / self.yesterday_directory.name
        self.today_directory = Path(self.original_root_directory.name) / self.today_directory.name

    def pre_run(self):
        """
        create backups
        run the test binary on yesterday and today to check the properties

        :return:
        """
        print("Pre-run test")
        ytd_ret = self.test_ytd()
        self.copy_to_tmp(force=True)
        today_ret = self.test_today()
        self.copy_to_tmp(force=True)



        assert ytd_ret == 0, "Yesterday's code did not work?"
        assert today_ret == 1, "Today's code did not break?"

    def test_ytd(self):
        """
        Test yesterday's directory. Patches are applied to yesterday, so it will be used to test patched code
        Reset tmp files
        :return:
        """
        ytd_run = subprocess.run([str(self.test_binary.resolve()), str(self.yesterday_directory.resolve())])
        print("yesterday test return code", ytd_run.returncode)

        return ytd_run.returncode

    def test_today(self):
        today_run = subprocess.run([str(self.test_binary.resolve()), str(self.today_directory.resolve())])
        print("today test return code", today_run.returncode)

        return today_run.returncode

    def debug(self, algo=1, create_patch=True, print_results=True):
        """
        Delta debug.

        :param algo: Pick which algorithm you want to use, 1 or 2.
        :param create_patch: If you want to create a patch for the minimal set of failure inducing changes
        :param print_results: If you want to print out the results of the incremental patches in the minimal set
        of failure inducing changes
        :return:
        """
        self.big_patch_path = Path("bigpatch")
        with self.big_patch_path.open("w+") as big_patch:
            # create a diff patch
            subprocess.run(["diff", "-up", "-r", str(self.yesterday_directory), str(self.today_directory)],
                           stdout=big_patch)

            # splitpatch into hunks
            subprocess.run(["splitpatch", str(self.big_patch_path)], stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL)

        incrementals = []
        for p in self.tmp_dir.iterdir():
            if p.suffix == ".patch":
                incrementals.append(p)

        # if self.debug_flag:
        #     print(incrementals)

        if algo == 1:
            minimal_patches = self.algo1(incrementals, set())
        elif algo == 2:
            minimal_patches = self.algo2(incrementals, set(), 2)
        else:
            raise Exception("Your algorithm choice is not supported, only 1 or 2")

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
        """
        Algorithm 1 presented in the paper. Does not handle inconsistencies.

        :param patches: The set of patches. c in paper.
        :param fixed: The fixed changes that will be applied to the subset. r in paper.
        :return: A possibly smaller set of patches.
        """
        if len(patches) == 1:
            return patches

        print("Algorithm 1 patches set length: " + str(len(patches)) + ", fixed length: " + str(len(fixed)))

        # split the patches into two sets
        c1 = set(list(patches)[0:len(patches) // 2])
        c2 = set(list(patches)[len(patches) // 2:])

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
                cis.append(set(list(patches)[i * set_size:]))
            else:
                cis.append(set(list(patches)[i * set_size: (i + 1) * set_size]))

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

        #
        #
        #     if tis[0] == 0 and tis[1] == 0:
        #         return self.algo2(cis[0], cis[1] | fixed, 2) | self.algo2(cis[1], cis[0] | fixed, 2)
        # else:
        #     complements = [set(patches) - ci for ci in cis]
        #     for i, complement in enumerate(complements):
        #         ci = cis[i]
        #         if self.test_patches(complement) == 0 and tis[i] == 0:
        #             return self.algo2(ci, complement | fixed, 2) | self.algo2(complement, ci | fixed, 2)

        # case 4
        # if n == 2:
        #     if tis[0] == -1 and tis[1] == 0:
        #         return self.algo2(cis[0], cis[1] | fixed, 2)
        #     elif tis[0] == 0 and tis[1] == -1:
        #         return self.algo2(cis[1], cis[0]|fixed, 2)
        # else:

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

        n_prime = min(len(c_prime), 2 * n)

        # for real
        if n < len(patches):
            return self.algo2(c_prime, r_prime, n_prime)

        # case 6
        return c_prime

    def test_patches(self, patches):
        """
        Apply patches and run the test
        path -R is not a good idea, because the tested program might modify the files, therefore
        some file changes are not reflected in created patches. We do complete backups.
        :param patches:
        :return:
        """
        # for some unknown reason, the patches are applied to the today's directory, not yesterday.
        # so we have to revert today, so it's the same as yesterday, then apply patches on today, then test today
        with self.big_patch_path.open('r') as big_patch:
            if self.debug_flag:
                subprocess.run(["patch", "-f", "-R", "-p0"], stdin=big_patch)
            else:
                subprocess.run(["patch", "-f", "-R", "-p0"], stdin=big_patch, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        ret_code = self.test_today()
        assert ret_code == 0

        # combine patches together
        plist = [str(p) for p in patches]
        combined_path = self.tmp_dir / "combined"
        if combined_path.exists():
            os.remove(combined_path)

        if len(plist) == 0:
            pass
        else:
            if len(plist) == 1:
                shutil.copy(plist[0], combined_path)
            else:
                with combined_path.open("w") as combined:
                    subprocess.run(["combinediff", "-q"] + plist[0:2], stdout=combined)
                for patch in plist[2:]:
                    second_combined_path = self.tmp_dir / "combined2"
                    with second_combined_path.open("w") as second_combined:
                        subprocess.run(["combinediff", "-q", str(patch), str(combined_path)], stdout=second_combined)
                        shutil.copy(str(second_combined_path), str(combined_path))

            # patch the files
            with combined_path.open("r") as combined:
                if self.debug_flag:
                    subprocess.run(["patch", "-f", "-p0"], stdin=combined)
                else:
                    subprocess.run(["patch", "-f", "-p0"], stdin=combined, stdout=subprocess.DEVNULL)

        # apply, test ytd, and revert
        ret_code = self.test_today()
        self.copy_to_tmp(force=True)

        # with combined_path.open("r") as combined:
        #     # with combined_path.open("r") as combined:
        #     if self.debug_flag:
        #         subprocess.run(["patch", "-R", "-p0", "-d/"], stdin=combined)
        #     else:
        #         subprocess.run(["patch", "-R", "-p0", "-d/"], stdin=combined, stdout=subprocess.DEVNULL)

        return ret_code

    def create_tmp_folder(self):
        this_path = pathlib.Path(__file__).parent.absolute()
        self.tmp_dir = this_path / "tmp"
        if self.tmp_dir.exists() and self.tmp_dir.is_dir():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir()


def demo():
    tb = "checker_test.py"
    yd = "demodir/yesterday"
    td = "demodir/today"
    root = "demodir"

    delta = Delta(tb, root, yd, td)
    delta.pre_run()
    delta.debug(algo=1)


def quick_dgd_demo():
    tb = "dbg_tester.py"
    yd = "dbg/find14"
    td = "dbg/find6"
    root = "dbg"

    delta = Delta(tb, root, yd, td)
    delta.pre_run()
    delta.debug(algo=1)


def quick_main():
    # relative paths work
    tb = "test_dummy.py"
    yd = "patches/expcp"
    td = "patches/expcp2"
    root = "patches"
    # yd = "/home/jasonhu/Desktop/rootdir/today"
    # td = "/home/jasonhu/Desktop/rootdir/yesterday"
    # root = "/home/jasonhu/Desktop/rootdir"

    delta = Delta(tb, root, yd, td)
    delta.pre_run()
    delta.debug(algo=1)


def main():
    tb = input("Please enter the test binary directory\n")
    root = input("Please enter the root directory where your yesterday and today directories reside\n")
    yd = input("Please enter the yesterday directory where your program works\n")
    td = input("Please enter the today directory where your program does not work :(\n")

    delta = Delta(tb, root, yd, td)
    delta.pre_run()
    delta.debug()


if __name__ == '__main__':
    quick_dgd_demo()
