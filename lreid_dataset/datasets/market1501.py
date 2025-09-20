from __future__ import division, print_function, absolute_import
import os
import copy
from lreid_dataset.incremental_datasets import IncrementalPersonReIDSamples
# from reid.utils.serialization import read_json, write_json
# from lreid_dataset.datasets import ImageDataset
import re
import glob
import os.path as osp
import warnings

from lreid_dataset.data.base_dataset import BaseImageDataset
import collections

class IncrementalSamples4market(BaseImageDataset):
    dataset_dir = ''

    def __init__(self, root,cam_filter=None, verbose=True, **kwargs):
        super(IncrementalSamples4market, self).__init__()
        self.global_pid2label = {}
        self.next_pid_label = 0
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.cam_filter = cam_filter

        self._check_before_run()
        self.get_global_lable(self.train_dir) # 建立全局映射
        if cam_filter is not None:
            train,cam_num_class = self._process_cam_dir(self.train_dir,cam_filter,relabel=False) # 返回datasets[camid] = {(img,pid,camid)}
            self.num_classes = cam_num_class
        else:
            train = self._process_dir(self.train_dir, relabel=False) # 返回dataset = {(img,pid,camid)}
            self.num_classes,_,_ = self.get_imagedata_info(train)

        query = self._process_dir(self.query_dir, relabel=False) 
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        
        self.train = train
            
        if verbose: 
            print("=> Market1501 loaded")
            print(f"[DEBUG] cam_filter={cam_filter},返回 {len(train)} 张图, 相机ID集合={set([c for _, _, c in train])},ID类别={self.num_classes}")

            self.print_dataset_statistics(train, query, gallery)


        self.query = query
        self.gallery = gallery


        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
    
    def get_global_lable(self,train_dir):
        cam_num = 6
        img_paths = glob.glob(osp.join(train_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        for cam_idx in range(cam_num):
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid <= 0:
                    continue  # junk images are just ignored
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 1 <= camid <= 6
                camid -= 1  # index starts from 0
                if camid!=cam_idx:
                    continue
                # 映射 pid -> 全局 id
                if pid not in self.global_pid2label:
                    self.global_pid2label[pid] = self.next_pid_label
                    self.next_pid_label += 1

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid <= 0:
                continue  # junk images are just ignored
            if pid in self.global_pid2label:
                global_pid = self.global_pid2label[pid]
            else:
                global_pid = pid
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            dataset.append((img_path, global_pid, camid))

        return dataset



    def _process_train_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
           
        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            pid = pid*6+camid
            dataset.append((img_path, pid, camid))

        return dataset


    def _process_cam_dir(self, dir_path,cam_filter, relabel=False):
        cam_num = 6
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        datasets = collections.defaultdict(list)
        cam_num_classes = [] # 统计每个相机的类别数

        for cam_idx in range(cam_num):
            dataset = []
            pids = set()
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid <= 0:
                    continue  # junk images are just ignored
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 1 <= camid <= 6
                camid -= 1  # index starts from 0
                if camid!=cam_idx:
                    continue
                global_pid = self.global_pid2label[pid]

                dataset.append((img_path, global_pid, camid))
                pids.add(global_pid)

            datasets[cam_idx] = dataset
            cam_num_classes.append(len(pids))

        return datasets[cam_filter],cam_num_classes[cam_filter]

