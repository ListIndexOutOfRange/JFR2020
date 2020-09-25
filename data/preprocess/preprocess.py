import os
import numpy as np
from tqdm import tqdm

import json
import nibabel as nib

from skimage.segmentation import clear_border

from data.preprocess import Patient


"""" This file does the following: 
    
    1. Create two folders named 'scans' and 'masks' in OUTPUT_DIR.

    2. From INPUT_DIR and MAX_DEPTH, construct a list of tuple (json_path, nifti_path),
       with the files being chosen as both annotated and not to deep. 
    
    3. STEP 1: For each selected scan, construct a mask by doing a logical AND between a mask obtained 
       by thresholding (130 HU) and one obtained by dilating around annotations coordinates.
       Store the mask in OUTPUT_DIR/masks/ as .npy.

    4. STEP 2: For each selected scan and its associated mask, crop in 3d based on mean intensity values.
       Store the cropped scans in OUTPUT_DIR/scans/ and the cropped scans in OUTPUT_DIR/masks/ as .npy.

    5. STEP 3: For each selected scan and its associated mask, create cubes of fixed size with a specified 
               depth, by z padding, z cutting, then xy cutting each z cutted cube.
     
    This 6 things are wrapped into one function.
    This fonction can take a list of steps to execute.
"""



class Preprocess:

    def __init__(self, input_dir, output_dir, max_depth):
        self.input_dir      = input_dir
        self.output_dir     = output_dir
        # dataset_paths is a list of tuple [(json_path, nifti_path)]
        self.dataset_paths  = self.get_dataset_paths(max_depth)
        # output_paths is a list of tuple [(scan_array_path, mask_array_path)]
        self.output_paths   = self.get_output_paths()

    def get_good_json_paths(self):
        """ Some jsons have class annotations but no localisations info.
            We can't use them for segmentation, hence we drop their paths.
        """
        all_json_files = list(filter(lambda x: x.endswith(".json"), os.listdir(self.input_dir)))
        all_json_paths = list(map(lambda x: os.path.join(self.input_dir, x), all_json_files))
        good_json_paths = []
        for json_path in all_json_paths:
            with open(json_path) as json_file:
                data = json.load(json_file)
                for key in data.keys():
                    if key.isdigit() and len(data[key][0]['labels']) > 0:
                        good_json_paths.append(json_path)
        return good_json_paths

    def select_one_nifti_path(self, json_path, all_paths, max_depth):
        """ One scan is sometimes associated with several nifti images.
            We select the deepest one being smaller than max_depth.
        """
        nifti_candidates = list(filter(lambda x: x.startswith(json_path[:-5]) and x.endswith(".nii.gz"), all_paths))
        max_z = 1
        final_nifti_path = nifti_candidates[0]
        for nifti_path in nifti_candidates:
            z = nib.load(nifti_path).header.get_data_shape()[2]
            if max_z < z < max_depth:
                max_z = nib.load(nifti_path).header.get_data_shape()[2]
                final_nifti_path = nifti_path
        return final_nifti_path

    def get_dataset_paths(self, max_depth):
        """ Takes an input dir and return a list of tuple (json_path, nifti_path). """
        dataset_paths = []
        all_paths = list(map(lambda x: os.path.join(self.input_dir, x), os.listdir(self.input_dir)))
        good_json_paths = self.get_good_json_paths()
        for json_path in good_json_paths:
            nifti_path = self.select_one_nifti_path(json_path, all_paths, max_depth)
            dataset_paths.append((json_path,nifti_path))
        return dataset_paths

    def prepare_output_folders(self):
        """ Create output folder and subfolders if needed. """
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        preprocessed_scans_dir = os.path.join(self.output_dir, 'scans')
        preprocessed_masks_dir = os.path.join(self.output_dir, 'masks')  
        if not os.path.isdir(preprocessed_scans_dir):
            os.mkdir(preprocessed_scans_dir)
        if not os.path.isdir(preprocessed_masks_dir):
            os.mkdir(preprocessed_masks_dir)

    def get_output_paths(self):
        """ Generates a list of tuple [(preprocessed_scan_path, preprocessed_mask_path)]"""
        output_paths = []
        for paths in self.dataset_paths:
            scan_name = paths[1][len(self.input_dir):-7]
            output_scan_path = os.path.join(self.output_dir, 'scans/', scan_name + '.npy')
            output_mask_path = os.path.join(self.output_dir, 'masks/', scan_name + '.npy')
            output_paths.append((output_scan_path, output_mask_path))
        return output_paths

    def step1(self, cube_side):
        """ Make and optionnaly store masks (uncropped) in OUTPUT_DIR/masks/ ."""
        for i in tqdm(range(len(self.dataset_paths))):
            json_path, nifti_path = self.dataset_paths[i] 
            patient = Patient(json_path, nifti_path)
            output_mask_path = self.output_paths[i][1]
            patient.make_mask(cube_side)
            patient.save_mask(output_mask_path)

    def step2(self, factor, margin):
        """ Crop scans a masks and optionnaly store the outputs respectively in 
            OUTPUT_DIR/scans and OUTPUT_DIR/masks/ .
        """
        for i in tqdm(range(len(self.dataset_paths))):
            output_scan_path, mask_path = self.output_paths[i][0], self.output_paths[i][1]
            patient = Patient(self.dataset_paths[i][0], self.dataset_paths[i][1])
            patient.load_mask(mask_path)
            patient.rescale('up') 
            patient.crop_3d(factor, margin)
            patient.rescale('down') 
            patient.save_scan(output_scan_path)
            patient.save_mask(mask_path)

    def step3(self, target_depth):
        """ From a couple (scan,mask) of same size (s,s,d)
            generates several cubes of size (s//2,s//2, target_depth)
            by doing:
            1. z padding
            2. z cutting of target_depth
            3. x,y cutting in four
        """
        for i in tqdm(range(len(self.dataset_paths))):
            output_scan_path, mask_path = self.output_paths[i][0], self.output_paths[i][1]
            patient = Patient(self.dataset_paths[i][0], self.dataset_paths[i][1])
            patient.load(mask_path)
            patient.cut(target_depth)
            patient.save_cutted_scans(output_scan_path[:-4])
            patient.save_cutted_masks(mask_path[:-4])

    #TODO: for now we pass input_dir and ouput_dir to do augment after preprocess but it
    #      shouldn't be this way.
    def augment(self, input_dir, output_dir, augment_factor, augment_proba):
        """ From each couple (scan,mask) generates as many augmented
            new couples as augment_factor.
        """
        self.output_dir = output_dir
        self.prepare_output_folders()
        scan_root = os.path.join(input_dir, "scans/")
        mask_root = os.path.join(input_dir, "masks/")
        scan_list = os.listdir(scan_root)
        mask_list = os.listdir(mask_root)
        assert len(scan_list) == len(mask_list)
        dataset_length = len(scan_list)
        for i in tqdm(range(dataset_length)):
            patient = Patient()
            patient.load_scan(os.path.join(scan_root, scan_list[i]))
            patient.load_mask(os.path.join(mask_root, mask_list[i]))
            for j in range(augment_factor):
                augmented_scan_name = f"{scan_list[i][:-4]}_{j}"
                augmented_mask_name = f"{mask_list[i][:-4]}_{j}"
                augmented_scan_path = os.path.join(output_dir, "scans/", augmented_scan_name)
                augmented_mask_path = os.path.join(output_dir, "masks/", augmented_mask_name)
                patient.augment(augment_proba)
                patient.save_augmented_scan(augmented_scan_path)
                patient.save_augmented_mask(augmented_mask_path)

    #TODO: this function is bugged because augment applies on voxel_arrays instead of cutted cubes
    def fast_all_steps(self, cube_side=10, factor=2, margin=5, target_depth=64, augment_factor=10, augment_proba=0.7):
        """ Performs all steps (1,2,3 + augment) faster by avoiding intermediate save/load. """
        self.prepare_output_folders()
        for i in tqdm(range(len(self.dataset_paths))):
            json_path, nifti_path = self.dataset_paths[i] 
            output_scan_path, mask_path = self.output_paths[i][0], self.output_paths[i][1]
            patient = Patient(json_path, nifti_path)
            patient.make_mask(cube_side)
            patient.rescale('up') 
            patient.crop_3d(factor, margin)
            patient.rescale('down') 
            patient.cut(target_depth)
            for j in range(augment_factor):
                augmented_scan_path = f"{output_scan_path[:-4]}_{j}"
                augmented_mask_path = f"{mask_path[:-4]}_{j}"
                patient.augment(augment_proba)
                patient.save_augmented_scan(augmented_scan_path)
                patient.save_augmented_mask(augmented_mask_path)

    def preprocess_dataset(self, steps, cube_side=10, factor=2, margin=5, target_depth=64, augment_factor=10, augment_proba=0.7):
        self.prepare_output_folders()
        if 1 in steps:
            print("STEP 1: Creating Masks...")
            self.step1(cube_side)
        if 2 in steps:
            print("STEP 2: Cropping Scans & Masks...")
            self.step2(factor, margin)
        if 3 in steps:
            print("STEP 3: Cutting Scans & Masks...")
            self.step3(target_depth)
        if 'augment' in steps:
            self.augment(augment_factor, augment_proba)

    def test(self, cube_side=10, factor=2, margin=5, target_depth=64, augment_factor=10, augment_proba=0.7):
        self.prepare_output_folders()
        for i in tqdm(range(len(self.dataset_paths))):
            json_path, nifti_path = self.dataset_paths[i] 
            output_scan_path, mask_path = self.output_paths[i][0], self.output_paths[i][1]
            patient = Patient(json_path, nifti_path)
            patient.make_mask(cube_side)
            patient.rescale('up') 
            patient.crop_3d(factor, margin)
            patient.rescale('down') 
            patient.cut(target_depth)
            patient.save_cutted_scans(output_scan_path[:-4])
            patient.save_cutted_masks(mask_path[:-4])