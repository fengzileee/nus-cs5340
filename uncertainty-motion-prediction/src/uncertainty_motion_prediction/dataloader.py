"""A module contains utilities and classes for evaluation."""
import os

from tqdm import tqdm, tqdm_pandas

tqdm.pandas()

from toolkit.loaders.loader_eth import load_eth
from toolkit.loaders.loader_crowds import load_crowds
from toolkit.core.trajdataset import TrajDataset, merge_datasets


class Dataloader():
    def __init__(self, opentraj_root="../OpenTraj/"):
        self._lut =[
            ["eth-univ", load_eth, os.path.join(opentraj_root, "datasets/ETH/seq_eth/obsmat.txt")],
            ["eth-hotel",load_eth, os.path.join(opentraj_root, "datasets/ETH/seq_hotel/obsmat.txt")],
            ["ucy-zara1",load_eth, os.path.join(opentraj_root, "datasets/UCY/zara01/obsmat.txt")],
            ["ucy-zara2",load_eth, os.path.join(opentraj_root, "datasets/UCY/zara02/obsmat.txt")],
            ["ucy-univ3",load_eth, os.path.join(opentraj_root, "datasets/UCY/students03/obsmat.txt")],
            #["ucy-zara", self.load_ucy_zara, "unused" ],
            #["ucy-univ", self.load_ucy_univ, "unused" ],
        ]
        self.opentraj_root = opentraj_root
        self.total_num = len(self._lut);

    def load(self, index):
        if index >= self.total_num:
            print("Out of bound, available datasets:")
            for i in range(self.total_num):
                print(i,self._lut[i][0])
            return [];
        return self._lut[index][1](self._lut[index][2])

    def get_key(self):
        return [self._lut[i][0]  for i in range(len(self._lut))]

    def load_ucy_zara(self,path):
        zara01_dir = os.path.join(self.opentraj_root, 'datasets/UCY/zara01')
        zara02_dir = os.path.join(self.opentraj_root, 'datasets/UCY/zara02')
        zara03_dir = os.path.join(self.opentraj_root, 'datasets/UCY/zara03')
        zara_01_ds = load_crowds(zara01_dir + '/annotation.vsp',
                             homog_file=zara01_dir + '/H.txt',
                             scene_id='1', use_kalman=True)
        zara_02_ds = load_crowds(zara02_dir + '/annotation.vsp',
                             homog_file=zara02_dir + '/H.txt',
                             scene_id='2', use_kalman=True)
        zara_03_ds = load_crowds(zara03_dir + '/annotation.vsp',
                             homog_file=zara03_dir + '/H.txt',
                             scene_id='3', use_kalman=True)
        return merge_datasets([zara_01_ds, zara_02_ds, zara_03_ds])

    def load_ucy_univ(self,path):
        st001_dir = os.path.join(self.opentraj_root, 'datasets/UCY/students01')
        st003_dir = os.path.join(self.opentraj_root, 'datasets/UCY/students03')
        uni_ex_dir = os.path.join(self.opentraj_root, 'datasets/UCY/uni_examples')
        #st001_ds = load_Crowds(st001_dir + '/students001.txt',homog_file=st001_dir + '/H.txt',scene_id='1',use_kalman=True)

        st001_ds = load_crowds(st001_dir + '/annotation.vsp',
                               homog_file=st003_dir + '/H.txt',
                               scene_id='st001', use_kalman=True)

        st003_ds = load_crowds(st003_dir + '/annotation.vsp',
                               homog_file=st003_dir + '/H.txt',
                               scene_id='st003', use_kalman=True)

        uni_ex_ds = load_crowds(uni_ex_dir + '/annotation.vsp',
                                homog_file=st003_dir + '/H.txt',
                                scene_id='uni-ex', use_kalman=True)
        return merge_datasets([st001_ds, st003_ds, uni_ex_ds])

