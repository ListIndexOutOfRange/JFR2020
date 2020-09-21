import os
import math
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import json
import nibabel as nib

from skimage.segmentation import clear_border


class Patient:


	def __init__(self, json_path, nifti_path):
		
		self.scan_voxel_array = self.get_voxel_array(nifti_path)
		assert self.scan_voxel_array.shape[0] == self.scan_voxel_array.shape[1]
		self.side_length      = self.scan_voxel_array.shape[0]
		self.nb_slices		  = self.scan_voxel_array.shape[2]
		self.annotations      = self.get_annotations(json_path)
		self.annotations_mask = np.zeros(shape=self.scan_voxel_array.shape)
		self.threshold_mask   = np.zeros(shape=self.scan_voxel_array.shape)
		self.mask_voxel_array = None # will be created by a call to self.make_mask()
		self.mean_intensity_stack = self.scan_voxel_array.mean(axis=(0,1))
		self.intensity_min, self.intensity_max = 0., 0.


	@property
	def rescaling(self):
	    # Some scans have offsetted pixel values.
	    if np.max(self.mean_intensity_stack) < -600:
	        return 604
	    if np.min(self.mean_intensity_stack) > 0:
	        return -420
	    return 0

	def rescale(self, mode='up'): # can be up or down
		if mode == 'up':
			self.scan_voxel_array += self.rescaling
		if mode == 'down':
			self.scan_voxel_array -= self.rescaling
		self.mean_intensity_stack = self.scan_voxel_array.mean(axis=(0,1))


	def get_voxel_array(self, nifti_path):
	    """ From a nifti path returns a 3D voxel array. """
	    scan = nib.load(nifti_path)
	    # The nifti data are 4ds, with one being of size 1 or 2. 
	    array_4d = scan.get_fdata()
	    if array_4d.shape[3] == 1:
	        return np.squeeze(array_4d, axis=3)
	    else:
	        return array_4d[:,:,:,0]


	def get_annotations(self, json_path):
	    """ From a json paths returns a list of triplets [(x,y,z)]
	        Each of this triplet is an annotation.
	    """
	    with open(json_path) as json_file:
	        data = json.load(json_file)
	        annotations = []
	        for key in data.keys():
	            if key.isdigit():
	                data['annotations'] = data.pop(key)
	        for annotation in data['annotations']:
	            z_coord = annotation['instance']
	            for label in annotation['labels']:
	                x_coord, y_coord = label['x'], label['y']
	                annotations.append((x_coord, y_coord, z_coord))
	    return annotations


	def make_threshold_mask(self):
	    for slice_index in range(self.nb_slices):
	        self.threshold_mask[:,:,slice_index] = clear_border(self.scan_voxel_array[:,:,slice_index] > 130)


	def make_one_annotation_mask(self, annotation, cube_side):
	    """ Make a mask from a triplet (x,y,z) by
	        putting white voxels in a cube of side 10 arround the annotated pixel.
	    """
	    x,y,z = annotation
	    x,y = int(x), int(y)
	    max_x, max_y, max_z = self.annotations_mask.shape
	    z_range = range(max(z-cube_side//2, 0), min(z+cube_side//2,max_z))
	    y_range = range(max(y-cube_side//2, 0), min(y+cube_side//2,max_y))
	    x_range = range(max(x-cube_side//2, 0), min(x+cube_side//2,max_x))
	    for z_idx in z_range:
	        for y_idx in y_range:
	            for x_idx in x_range: 
	                self.annotations_mask[y_idx,x_idx,z_idx] = 1


	def make_annotations_mask(self, cube_side):
		for annotation in self.annotations:
			self.make_one_annotation_mask(annotation, cube_side)


	def make_mask(self, cube_side):
		""" Logical AND between the annotation mask and the threshold mask. """
		self.make_threshold_mask()
		self.make_annotations_mask(cube_side)
		self.mask_voxel_array = np.logical_and(self.threshold_mask, self.annotations_mask)
	

	def is_intensity_regular(self, slice_index, intensity_min, intensity_max):
		return (intensity_min <= self.mean_intensity_stack[slice_index] <= intensity_max)

	
	def delete_outliers(self, intensity_min, intensity_max):
	    """ Remove slices that have an mean intensity lower than -600 or greater than -150 """
	    slice_index = 0
	    while slice_index < self.nb_slices and not self.is_intensity_regular(slice_index, intensity_min, intensity_max):
	        slice_index += 1
	    min_z = slice_index
	    while slice_index < self.nb_slices and self.is_intensity_regular(slice_index, intensity_min, intensity_max):
	        slice_index += 1
	    max_z = slice_index-1
	    self.scan_voxel_array = self.scan_voxel_array[:,:,min_z:max_z]
	    self.mask_voxel_array = self.mask_voxel_array[:,:,min_z:max_z]
	    self.nb_slices = max_z - min_z


	def get_intensity_range(self, factor):
		array_intensity_mean = np.mean(self.mean_intensity_stack)
		array_intensity_std  = np.std(self.mean_intensity_stack)
		intensity_min = array_intensity_mean - factor*array_intensity_std
		intensity_max = array_intensity_mean + factor*array_intensity_std
		return intensity_min, intensity_max


	def crop_z(self, factor):
	    """ Get the z range to keep by calculating mean and std of intensity values of each slice. """
	    self.delete_outliers(-600, -150)
	    intensity_min, intensity_max = self.get_intensity_range(factor) 
	    self.delete_outliers(intensity_min, intensity_max)
	    

	def crop_xy(self):
		""" Reduce slice size by keeping a centered square of size old_shape/sqrt(2) """
		new_side_length = int(self.side_length/math.sqrt(2))
		center_index = self.side_length//2
		offset = center_index - new_side_length//2
		cropped_scan = np.ndarray(shape=(new_side_length, new_side_length, self.nb_slices))
		cropped_mask = np.ndarray(shape=(new_side_length, new_side_length, self.nb_slices))
		for slice_index in range(self.nb_slices):
			for x in range(new_side_length):
				for y in range(new_side_length):
					cropped_scan[x,y,slice_index] = self.scan_voxel_array[offset+x,offset+y,slice_index]
					cropped_mask[x,y,slice_index] = self.mask_voxel_array[offset+x,offset+y,slice_index]
		self.scan_voxel_array = cropped_scan
		self.mask_voxel_array = cropped_mask


	def crop_3d(self, factor):
	    self.crop_z(factor)
	    self.crop_xy()


	def save_scan(self, scan_path):
		np.save(scan_path, self.scan_voxel_array)


	def save_mask(self, mask_path):
		np.save(mask_path, self.mask_voxel_array)


	def load_mask(self, path):
		self.mask_voxel_array = np.load(path)