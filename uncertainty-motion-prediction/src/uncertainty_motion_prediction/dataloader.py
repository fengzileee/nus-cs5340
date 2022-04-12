"""A module contains utilities and classes for evaluation."""
import os
import numpy as np
from tqdm import tqdm, tqdm_pandas

tqdm.pandas()

from toolkit.loaders.loader_eth import load_eth
from toolkit.loaders.loader_crowds import load_crowds
from toolkit.loaders.loader_kitti import load_kitti
from toolkit.loaders.loader_wildtrack import load_wildtrack
from toolkit.core.trajdataset import TrajDataset, merge_datasets


class Dataloader():
    def __init__(self, opentraj_root="../OpenTraj/"):
        self._lut =[
            ["eth-univ", self.load_eth_dataset, os.path.join(opentraj_root, "datasets/ETH/seq_eth/obsmat.txt")],
            ["eth-hotel",self.load_eth_dataset, os.path.join(opentraj_root, "datasets/ETH/seq_hotel/obsmat.txt")],
            #["wildtrack",self.load_wild_track, "unused"],
            ["ucy-zara1",self.load_eth_dataset, os.path.join(opentraj_root, "datasets/UCY/zara01/obsmat.txt")],
            ["ucy-zara2",self.load_eth_dataset, os.path.join(opentraj_root, "datasets/UCY/zara02/obsmat.txt")],
            ["ucy-univ3",self.load_eth_dataset, os.path.join(opentraj_root, "datasets/UCY/students03/obsmat.txt")],
            ["ucy-zara", self.load_ucy_zara, "unused" ],
            #["kitti",    self.load_kitti_dataset, os.path.join(opentraj_root, 'datasets/KITTI/data')],
        ]
        self.opentraj_root = opentraj_root
        self.total_num = len(self._lut);

    def load(self, index):
        if index >= self.total_num:
            print("Out of bound, available datasets:")
            for i in range(self.total_num):
                print(i,self._lut[i][0])
            return [];
        return self._lut[index][1](self._lut[index][2], self._lut[index][0])

    def get_key(self):
        return [self._lut[i][0]  for i in range(len(self._lut))]

    def load_eth_dataset(self,path, scene_id):
        dataset = load_eth(path)
        dataset.data["scene_id"] = scene_id
        return dataset


    def load_ucy_zara(self,path, scene_id):
        zara01_dir = os.path.join(self.opentraj_root, 'datasets/UCY/zara01')
        zara02_dir = os.path.join(self.opentraj_root, 'datasets/UCY/zara02')
        zara03_dir = os.path.join(self.opentraj_root, 'datasets/UCY/zara03')
        zara_01_ds = load_crowds(zara01_dir + '/annotation.vsp',
                             homog_file=zara01_dir + '/H.txt',
                             scene_id='1', use_kalman=False)
        zara_02_ds = load_crowds(zara02_dir + '/annotation.vsp',
                             homog_file=zara02_dir + '/H.txt',
                             scene_id='2', use_kalman=False)
        zara_03_ds = load_crowds(zara03_dir + '/annotation.vsp',
                             homog_file=zara03_dir + '/H.txt',
                             scene_id='3', use_kalman=False)
        return merge_datasets([zara_01_ds, zara_02_ds, zara_03_ds])

    def load_ucy_univ(self,path, scene_id):
        st001_dir = os.path.join(self.opentraj_root, 'datasets/UCY/students01')
        st003_dir = os.path.join(self.opentraj_root, 'datasets/UCY/students03')
        uni_ex_dir = os.path.join(self.opentraj_root, 'datasets/UCY/uni_examples')
        #st001_ds = load_Crowds(st001_dir + '/students001.txt',homog_file=st001_dir + '/H.txt',scene_id='1',use_kalman=True)

        st001_ds = load_crowds(st001_dir + '/annotation.vsp',
                               homog_file=st003_dir + '/H.txt',
                               scene_id='st001', use_kalman=False)

        st003_ds = load_crowds(st003_dir + '/annotation.vsp',
                               homog_file=st003_dir + '/H.txt',
                               scene_id='st003', use_kalman=False)

        uni_ex_ds = load_crowds(uni_ex_dir + '/annotation.vsp',
                                homog_file=st003_dir + '/H.txt',
                                scene_id='uni-ex', use_kalman=False)
        return merge_datasets([st001_ds, st003_ds, uni_ex_ds])
    def load_wild_track(self, path, scene_id):
        wildtrack_root = os.path.join(self.opentraj_root, 'datasets/Wild-Track/annotations_positions')
        return load_wildtrack(wildtrack_root,
                                                scene_id=scene_id,
                                                use_kalman=False,
                                                sampling_rate=1)  # original_annot_framerate=2
    def load_kitti_dataset(self, path, scene_id):
        dataset = load_kitti(path, use_kalman=False, sampling_rate=4)  # FixMe: apparently original_fps = 2.5
        dataset.data["scene_id"] = scene_id
        return dataset


    def load_train_dataset(self, ratio = 0.7):
        self.all = merge_datasets([ self.load(i)  for i in range(self.total_num)])
        df = self.all.data.copy(True)

        df["unique"] = df["agent_id"].astype(str) + df["scene_id"]
        test = df.copy(True)

        label = df["unique"].unique()
        mask = np.random.binomial(1, ratio, len(label))
        train_label = []
        test_label = []
        for i in range(len(label)):
            if mask[i] == 1:
                train_label.append(label[i])
            else:
                test_label.append(label[i])

        df = df[df.unique.isin(train_label)]
        df = df.drop(columns=['unique'])

        self.train  =  TrajDataset()
        self.train.data = df
        self.train.title = 'train' + self.all.title


        test = test[test.unique.isin(test_label)]
        test = test.drop(columns=['unique'])


        self.test  =  TrajDataset()
        self.test.data =test
        self.test.title = 'test' + self.all.title

        return self.train

    def load_test_dataset(self):
        return self.test