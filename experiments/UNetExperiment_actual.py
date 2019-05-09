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
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment

from networks.RecursiveUNet import UNet
from loss_functions.dice_loss import SoftDiceLoss


class UNetExperiment(PytorchExperiment):
    """
    The UnetExperiment is inherited from the PytorchExperiment. It implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    It is optimized to work with the provided NumpyDataLoader.

    The basic life cycle of a UnetExperiment is the same s PytorchExperiment:

        setup()
        (--> Automatically restore values if a previous checkpoint is given)
        prepare()

        for epoch in n_epochs:
            train()
            validate()
            (--> save current checkpoint)

        end()
    """

    def setup(self):
        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        tr_keys = splits[self.config.fold]['train']
        val_keys = splits[self.config.fold]['val']
        test_keys = splits[self.config.fold]['test']

        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.train_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                              keys=tr_keys)
        self.val_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=False)
        self.test_data_loader = NumpyDataSet(self.config.data_test_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                             keys=test_keys, mode="test", do_reshuffle=False)
        self.model = UNet(num_classes=self.config.num_classes, in_channels=self.config.in_channels)

        self.model.to(self.device)

        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax for DICE Loss!
        self.ce_loss = torch.nn.CrossEntropyLoss()  # No softmax for CE Loss -> is implemented in torch!

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model"))

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        self.model.train()

        data = None
        batch_counter = 0
        for data_batch in self.train_data_loader:

            self.optimizer.zero_grad()

            # Shape of data_batch = [1, b, c, w, h]
            # Desired shape = [b, c, w, h]
            # Move data and target to the GPU
            data = data_batch['data'][0].float().to(self.device)
            target = data_batch['seg'][0].long().to(self.device)
            print("data shape : ",str(data.shape ,"target shape :",str(target.shape)))
            pred = self.model(data)
            pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

            loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
            #loss = self.ce_loss(pred, target.squeeze())
            #dice cofficent  = -1*diceloss 
            accuracy = -1*self.dice_loss(pred_softmax, target.squeeze()) 

            loss.backward()
            self.optimizer.step()

            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Epoch: {0} Loss: {1:.4f} Accuracy : {1:.4f}'.format(self._epoch_idx, loss, accuracy))

                self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch + (batch_counter / self.train_data_loader.data_loader.num_batches))
                self.add_result(value=accuracy.item(), name='Train_Accuracy', tag='Accuracy', counter=epoch + (batch_counter / self.train_data_loader.data_loader.num_batches))
                self.clog.show_image_grid(data.float().cpu(), name="data", normalize=True, scale_each=True, n_iter=epoch)
                self.clog.show_image_grid(target.float().cpu(), name="mask", title="Mask", n_iter=epoch)
                self.clog.show_image_grid(torch.argmax(pred.cpu(), dim=1, keepdim=True), name="unt_argmax", title="Unet", n_iter=epoch)
                self.clog.show_image_grid(pred.cpu()[:, 1:2, ], name="unt", normalize=True, scale_each=True, n_iter=epoch)

            batch_counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'

    def validate(self, epoch):
        self.elog.print('VALIDATE')
        self.model.eval()

        data = None
        loss_list = []
        accuracy_List =[]
        val_batch_counter = 0  
        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch['seg'][0].long().to(self.device)

                pred = self.model(data)
                pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
                loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
                #loss = self.ce_loss(pred, target.squeeze()) 
                loss_list.append(loss.item())
                accuracy = -1*self.dice_loss(pred_softmax, target.squeeze()) 
                accuracy_List.append(accuracy)
                val_batch_counter +=1
        assert data is not None, 'data is None. Please check if your dataloader works properly'
        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f Accuracy: %.4f' % (self._epoch_idx, np.mean(loss_list),np.mean(accuracy_List)))
        
        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)
        self.add_result(value=accuracy.item(), name='Val_Accuracy', tag='Accuracy', counter=epoch + (val_batch_counter /         self.val_data_loader.data_loader.num_batches))
        self.clog.show_image_grid(data.float().cpu(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        self.clog.show_image_grid(target.float().cpu(), name="mask_val", title="Mask", n_iter=epoch)
        self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax_val", title="Unet", n_iter=epoch)
        self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_val", normalize=True, scale_each=True, n_iter=epoch)

    def test(self):
        # TODO
        print(' Inside the test method ')
        self.elog.print('----Inside Test method----')
        self.model.eval()
        testiter = 13 

        data = None
        loss_list = []
        accuracy_list = [] 
        with torch.no_grad():
            for data_batch in self.test_data_loader:
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch['seg'][0].long().to(self.device)

                pred = self.model(data)
                pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
                accuracy = -1*self.dice_loss(pred_softmax, target.squeeze())  
                loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
                 
                loss_list.append(loss.item())
                accuracy_list.append(accuracy) 
        assert data is not None, 'data is None. Please check if your dataloader works properly'
        #self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f accuracy: %.4f' % (self._epoch_idx, np.mean(loss_list),np.mean(accuracy_list)))

        self.add_result(value=np.mean(loss_list), name='Test_Loss', tag='Loss', counter=testiter)
        self.add_result(value=np.mean(accuracy_list), name='Test_Accuracy', tag='Accuracy', counter=testiter)

        self.clog.show_image_grid(data.float().cpu(), name="data_Test", normalize=True, scale_each=True, n_iter=testiter)
        self.clog.show_image_grid(target.float().cpu(), name="mask_Test", title="Mask", n_iter=testiter)
        self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax_Test", title="Unet", n_iter=testiter)
        self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_Test", normalize=True, scale_each=True, n_iter=testiter)


