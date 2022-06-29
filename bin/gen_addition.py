# -*- coding: utf-8 -*-
# last updated:<2022/06/05/Sun 10:34:12 from:tuttle-desktop>

import random
import txtutils

def main():
    for i in range(50000):
        x = random.randint(1, 1000)
        y = random.randint(1, 1000)
        z = x + y
        print(txtutils.normalize(" ".join(list(str(x)))+" + "+" ".join(list(str(y)))))
        print(txtutils.normalize(" ".join(list(str(z)))))
        print()

if __name__ == "__main__":
    main()
