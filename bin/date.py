# -*- coding: utf-8 -*-
# last updated:<2022/06/05/Sun 15:33:58 from:tuttle-desktop>

import txtutils

def main():
    with open("../data/date.txt") as f:
        txt = f.read().strip()

    for line in txt.split("\n"):
        input, output = line.split("_")
        input = list(txtutils.normalize(input.strip()))
        output = list(txtutils.normalize(output.strip()))
        print(" ".join(input))
        print(" ".join(output))
        print()


if __name__ == "__main__":
    main()
