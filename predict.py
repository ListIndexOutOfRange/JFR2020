#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:57:02 2020

@author: thomas
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from model import LightningModel


class PredictorCalci():
    
    
    def __init__(self, model_path = "", threshold = 0.5, device = 'cpu'):
        """
        

        Parameters
        ----------
        model_path : TYPE, optional
            pretrained model. The default is "/home/thomas/Bureau/jfr/JFR2020/lightning_logs/version_104/checkpoints/epoch=2.ckpt".
        threshold : TYPE, optional
           threshold for the sementation The default is 0.5.
        device : TYPE, optional
            DESCRIPTION. The default is 'cpu'.

        Returns
        -------
        None.

        """
        self.model = LightningModel.load_from_checkpoint(model_path).eval().to(device)
        self.threshold = threshold
        self.device = device 
        

    def get_class_from_volume(self, volume):
        if volume == 0: return 0
        if 1 <= volume <= 9: return 1
        if 10 <= volume <= 99: return 2
        if 100 <= volume <= 399: return 3
        return 4

    
    def predict_score(self, input_dir = "/home/thomas/Bureau/jfr/preprocessed/scans2/"):
        """
        

        Parameters
        ----------
        input_dir : directory with with cubes from patient. ex "/home/thomas/Bureau/jfr/sficv/"

        Returns
        -------
        dataframe with patientId, value score
        

        """
        
        
        ###
        #get file name
        ###
        
        list_file_name = glob.glob(input_dir + "*")
        list_patient_id = [list_file_name[i][len(input_dir):len(input_dir) + 24] for i in range(len(list_file_name))]
        dico_volume = {}
        for patient_id in list_patient_id:
            dico_volume[patient_id] = 0
        
        
        ###
        #compute the volume for each patient
        ###
        
        for index in tqdm(range(len(list_file_name))):
            scan = torch.tensor(np.load(list_file_name[index], allow_pickle=True)).unsqueeze(1)
            if torch.isnan(scan.max()):
                print("Nan detected at batch")
                scan[torch.isnan(scan)] = 0
            scan = scan.float().to(self.device)
            predicted_mask = self.model(scan)
            dico_volume[list_patient_id[index]] += torch.sum(predicted_mask >= self.threshold).to('cpu').item()
        ###
        #compute score for each patient
        ###
        list_score = []
        for k in  dico_volume.keys():
            list_score.append(self.get_class_from_volume(dico_volume[k]))
        
        d = {'examen_id': dico_volume.keys(), 'volume': dico_volume.values(), "score": list_score}
        result_df = pd.DataFrame(data=d)
        return result_df
        
        
        
     