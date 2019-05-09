#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import fnmatch
import random

import numpy as np

from batchgenerators.dataloading import SlimDataLoaderBase
from batchgenerators.dataloading.data_loader import DataLoader
from datasets.data_loader import MultiThreadedDataLoader
from .data_augmentation import get_transforms
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.crop_and_pad_augmentations import crop

def load_dataset(base_dir, pattern='*.npy', slice_offset=5, keys=None):
    fls = []
    files_len = []
    slices_ax = []

    for root, dirs, files in os.walk(base_dir):
        i = 0
        for filename in sorted(fnmatch.filter(files, pattern)):

            if keys is not None and filename[:-4] in keys:
                npy_file = os.path.join(root, filename)
                numpy_array = np.load(npy_file, mmap_mode="r")

                fls.append(npy_file)
                files_len.append(numpy_array.shape[1])

                slices_ax.extend([(i, j) for j in range(slice_offset, files_len[-1] - slice_offset)])

                i += 1

    return fls, files_len, slices_ax,


class NumpyDataSet(object):
    """
    TODO
    """
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000, seed=None, num_processes=8, num_cached_per_queue=8 * 4, target_size=128,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), do_reshuffle=True, keys=None):

        data_loader = NumpyDataLoader(base_dir=base_dir, mode=mode, batch_size=batch_size, num_batches=num_batches, seed=seed, file_pattern=file_pattern,
                                      input_slice=input_slice, label_slice=label_slice, keys=keys)
        #data_loader = BraTS2017DataLoader2D(train, batch_size, max_shape, 1)
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1

        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedDataLoader(data_loader, self.transforms, num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue, seeds=seed,
                                                 shuffle=do_reshuffle)
        self.augmenter.restart()

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.augmenter.renew()
        return self.augmenter

    def __next__(self):
        return next(self.augmenter)


class BraTS2017DataLoader2D(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234, return_incomplete=False,
                 shuffle=True, infinite=True):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)

        patch_size is the spatial size the retured batch will have

        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.patch_size = patch_size
        self.num_modalities = 4
        self.indices = list(range(len(data)))

    #@staticmethod
    #def load_patient(patient):
    #    return BraTS2017DataLoader3D.load_patient(patient)

    @staticmethod
    def load_patient(patient):
        data = np.load(patient + ".npy", mmap_mode="r")
        metadata = load_pickle(patient + ".pkl")
        return data, metadata

    def generate_train_batch(self,idx):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        #idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)

        metadata = []
        patient_names = []

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            patient_data, patient_metadata = self.load_patient(j)

            # patient data is a memmap. If we extract just one slice then just this one slice will be read from the
            # disk, so no worries!
            slice_idx = np.random.choice(patient_data.shape[1])
            patient_data = patient_data[:, slice_idx]

            # this will only pad patient_data if its shape is smaller than self.patch_size
            patient_data = pad_nd_image(patient_data, self.patch_size)

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            patient_data, patient_seg = crop(patient_data[:-1][None], patient_data[-1:][None], self.patch_size, crop_type="random")

            data[i] = patient_data[0]
            seg[i] = patient_seg[0]

            metadata.append(patient_metadata)
            patient_names.append(j)

        return {'data': data, 'seg':seg, 'metadata':metadata, 'names':patient_names}



class NumpyDataLoader(DataLoader):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 seed=None, file_pattern='*.npy', label_slice=1, input_slice=(0,), keys=None):

        self.files, self.file_len, self.slices = load_dataset(base_dir=base_dir, pattern=file_pattern, slice_offset=0, keys=keys, )
        print("files :",self.files,"  file_len :",self.file_len  ,"  slices :",self.slices)
        super(NumpyDataLoader, self).__init__(self.slices, batch_size, num_batches)

        self.batch_size = batch_size

        self.num_modalities = 4
        self.patch_size = (160, 160)

        self.use_next = False
        if mode == "train":
            self.use_next = False

        self.slice_idxs = list(range(0, len(self.slices)))
        print(" slice_idxs :",self.slice_idxs)
        self.data_len = len(self.slices)
        print(" data_len :",self.data_len)
        self.num_batches = min((self.data_len // self.batch_size)+10, num_batches)

        if isinstance(label_slice, int):
            label_slice = (label_slice,)
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.np_data = np.asarray(self.slices)

    def reshuffle(self):
        print("Reshuffle...")
        random.shuffle(self.slice_idxs)
        print("Initializing... this might take a while...")

    def generate_train_batch(self):
        open_arr = random.sample(self._data, self.batch_size)
        return self.get_data_from_array(open_arr)

    def __len__(self):
        n_items = min(self.data_len // self.batch_size, self.num_batches)
        return n_items

    @staticmethod
    def load_patient(patient):
        print("patient : ", patient)
        data = np.load(patient + ".npy", mmap_mode="r")
        metadata = load_pickle(patient + ".pkl")
        return data, metadata

    def generate_train_batch(self,idx):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        #idx = self.get_indices()
        #patients_for_batch = [self._data[i] for i in idx]
        
        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)

        metadata = []
        patient_names = []

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            patient_data, patient_metadata = self.load_patient(j)

            # patient data is a memmap. If we extract just one slice then just this one slice will be read from the
            # disk, so no worries!
            slice_idx = np.random.choice(patient_data.shape[1])
            patient_data = patient_data[:, slice_idx]

            # this will only pad patient_data if its shape is smaller than self.patch_size
            patient_data = pad_nd_image(patient_data, self.patch_size)

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            patient_data, patient_seg = crop(patient_data[:-1][None], patient_data[-1:][None], self.patch_size, crop_type="random")

            data[i] = patient_data[0]
            seg[i] = patient_seg[0]

            metadata.append(patient_metadata)
            patient_names.append(j)
        print("Returning data for brain generate train batch")
        return {'data': data, 'seg':seg, 'metadata':metadata, 'names':patient_names}

    def __getitem__(self, item):
        slice_idxs = self.slice_idxs
        data_len = len(self.slices)
        np_data = self.np_data

        if item > len(self):
            raise StopIteration()
        if (item * self.batch_size) == data_len:
            raise StopIteration()

        start_idx = (item * self.batch_size) % data_len
        stop_idx = ((item + 1) * self.batch_size) % data_len

        if ((item + 1) * self.batch_size) == data_len:
            stop_idx = data_len

        if stop_idx > start_idx:
            idxs = slice_idxs[start_idx:stop_idx]
        else:
            raise StopIteration()

        open_arr = np_data[idxs]
        #return self.generate_train_batch(idxs)  
        return self.get_data_from_array(open_arr)

    def get_data_from_array(self, open_array):
        data = []
        fnames = []
        slice_idxs = []
        labels = []

        for slice in open_array:
            fn_name = self.files[slice[0]]

            numpy_array = np.load(fn_name, mmap_mode="r")

            numpy_slice = numpy_array[:, slice[1]]
            print("numpy_array shape :",numpy_array.shape ,"numpy_slice shape :",numpy_slice.shape) 
            # this will only pad patient_data if its shape is smaller than self.patch_size
            patient_data = pad_nd_image(numpy_slice, self.patch_size)

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            patient_data, patient_seg = crop(patient_data[:-1][None], patient_data[-1:][None], self.patch_size, crop_type="random")
            
            dataslice = patient_data[0][0][None]
            dataslice = dataslice[0][None]
            segGt = patient_seg[0]
            #print("Before data shape :",dataslice.shape ,"sdpegGt :",segGt.shape) 
            #dataslice = dataslice[0] 
            print("After data shape :",dataslice.shape ,"segGt :",segGt.shape) 
            data.append(dataslice)   # 'None' keeps the dimension

            if self.label_slice is not None:
                labels.append(segGt)   # 'None' keeps the dimension

            fnames.append(self.files[slice[0]])
            slice_idxs.append(slice[1])
            
        ret_dict = {'data': data, 'fnames': fnames, 'slice_idxs': slice_idxs}
        if self.label_slice is not None:
            ret_dict['seg'] = labels
        print("data shape :",len(data) ,"labels :",len(labels)) 
        return ret_dict

    
