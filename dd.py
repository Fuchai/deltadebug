import subprocess

class Delta:
    def __init__(self, test_binary=None, yesterday_directory=None, today_directory=None):
        self.test_binary = test_binary
        self.yesterday_directory = yesterday_directory
        self.today_directory = today_directory

    def pre_run(self):
        """
        run the test binary on yesterday and today to check the properties

        :return:
        """
        ytd_run=subprocess.run([self.test_binary, self.yesterday_directory])
        today_run=subprocess.run([self.test_binary, self.today_directory])

        pass

    def debug(self, create_patch=True, print_results=True):
        """
        Runs algorithm one on the paper

        :return:
        """
        pass


def main():
    tb = input("Please enter the test binary directory\n")
    yd = input("Please enter the yesterday directory where your program works\n")
    td = input("Please enter the today directory where your program does not work :(\n")

    delta = Delta(tb, yd, td)
    delta.pre_run()
    delta.debug()


if __name__ == '__main__':
    main()
