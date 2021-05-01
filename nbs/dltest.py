from fastai.torch_basics import *
from fastai.data.load import *

class RandDL(DataLoader):
    def create_item(self, s):
        r = random.random()
        return r if r<0.95 else stop()

if __name__ == "__main__":
    # It can be reproduced in Linux by uncommenting this line
    # multiprocessing.set_start_method('spawn')
    print("start main ...")
    dl = RandDL(bs=4, num_workers=2, drop_last=True)
    print(L(dl).map(len))
