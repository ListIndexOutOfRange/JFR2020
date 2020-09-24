import os
import numpy as np
from tqdm import tqdm

import json
import nibabel as nib

from skimage.segmentation import clear_border

from utils.volumentations import *


class Patient:

	def __init__(self, json_path=None, nifti_path=None):
		# json and nifti paths can be none in case we just wanna 
		# init a Patient object to load voxel arrays later.
		if nifti_path is not None:
			self.scan_voxel_array = self.get_voxel_array(nifti_path)
			assert self.scan_voxel_array.shape[0] == self.scan_voxel_array.shape[1]
			self.side_length      = self.scan_voxel_array.shape[0]
			self.nb_slices		  = self.scan_voxel_array.shape[2]
			self.mean_intensity_stack = self.scan_voxel_array.mean(axis=(0,1))
			self.threshold_mask   = np.zeros(shape=self.scan_voxel_array.shape)
		if json_path is not None:
			self.annotations      = sorted(self.get_annotations(json_path),key=lambda x: x[2])
			self.annotations_mask = np.zeros(shape=self.scan_voxel_array.shape)
		self.mask_voxel_array = None # will be created by a call to self.make_mask()
		self.cutted_scans     = [] # contains cube of same size obtained by cutting original scan
		self.cutted_masks     = [] # contains cube of same size obtained by cutting original mask
		self.augmented_scan   = [] # will be filled by calling augment()
		self.augmented_mask   = [] # will be filled by calling augment()

	@property
	def rescaling(self):
	    # Some scans have offsetted pixel values.
	    if np.max(self.mean_intensity_stack) < -600: return 604
	    if np.min(self.mean_intensity_stack) > 0: return -420
	    return 0

	def rescale(self, mode='up'): # can be up or down
		if mode == 'up':
			self.scan_voxel_array += self.rescaling
		if mode == 'down':
			self.scan_voxel_array -= self.rescaling
		self.mean_intensity_stack = self.scan_voxel_array.mean(axis=(0,1))

# +-------------------------------------------------------------------------------------+ #
# |                                   		GET DATA	                                | #
# +-------------------------------------------------------------------------------------+ #
	
	def get_voxel_array(self, nifti_path):
		""" From a nifti path returns a 3D voxel array. """
		scan = nib.load(nifti_path)
		# The nifti data are 4ds, with one being of size 1 or 2. 
		array_4d = scan.get_fdata()
		if array_4d.shape[3] == 1:
			return np.squeeze(array_4d, axis=3)
		return array_4d[:,:,:,0]

	def get_annotations(self, json_path):
	    """ From a json paths returns a list of triplets [(x,y,z)].
	        Each of this triplet is an annotation.
	    """
	    with open(json_path) as json_file:
	        data = json.load(json_file)
	        annotations = []
			# rename key so that we can retrieve data from any json
	        for key in data.keys():
	            if key.isdigit():
	                data['annotations'] = data.pop(key)
	        for annotation in data['annotations']:
	            z_coord = annotation['instance']
	            for label in annotation['labels']:
	                x_coord, y_coord = label['x'], label['y']
	                annotations.append((x_coord, y_coord, z_coord))
	    return annotations

# +-------------------------------------------------------------------------------------+ #
# |                                   		MAKE MASK	                                | #
# +-------------------------------------------------------------------------------------+ #

	def make_threshold_mask(self):
	    for slice_index in range(self.nb_slices):
	        self.threshold_mask[:,:,slice_index] = clear_border(self.scan_voxel_array[:,:,slice_index] > 130)

	def make_one_annotation_mask(self, annotation, cube_side):
	    """ Make a mask from a triplet (x,y,z) by
	        putting white voxels in a cube of side cube_side arround the annotated pixel.
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

# +-------------------------------------------------------------------------------------+ #
# |                                   		CROP   		                                | #
# +-------------------------------------------------------------------------------------+ #

	def get_z_range_by_annotations(self, margin=5):
		""" Returns the lowest and highest slices with annotations plus a specified margin.
			This margin prevent annotations from being in the edge of a cropped cube.
		"""
		min_z = max(0, self.annotations[0][2]-margin)
		max_z = min(self.nb_slices, self.annotations[-1][2]+margin)
		return min_z, max_z
	
	def is_intensity_regular(self, slice_index, intensity_min, intensity_max):
		return (intensity_min <= self.mean_intensity_stack[slice_index] <= intensity_max)

	def get_z_range_by_intensity(self, intensity_min, intensity_max):
		""" Remove slices that have an mean intensity lower than intensity_min or greater than intensity_max """
		slice_index = 0
		while slice_index < self.nb_slices and not self.is_intensity_regular(slice_index, intensity_min, intensity_max):
			slice_index += 1
		min_z = slice_index
		while slice_index < self.nb_slices and self.is_intensity_regular(slice_index, intensity_min, intensity_max):
			slice_index += 1
		max_z = slice_index-1
		return min_z, max_z

	def get_intensity_range(self, factor):
		array_intensity_mean = np.mean(self.mean_intensity_stack)
		array_intensity_std  = np.std(self.mean_intensity_stack)
		intensity_min = array_intensity_mean - factor*array_intensity_std
		intensity_max = array_intensity_mean + factor*array_intensity_std
		return intensity_min, intensity_max
	
	def delete_outliers(self, intensity_min, intensity_max, margin, use_annotations=True):
		min_z_by_intensity, max_z_by_intensity = self.get_z_range_by_intensity(intensity_min, intensity_max)
		if use_annotations:
			min_z_by_annotations, max_z_by_annotations = self.get_z_range_by_annotations(margin)
			min_z = min(min_z_by_intensity, min_z_by_annotations)
			max_z = max(max_z_by_intensity, max_z_by_annotations)
		else:
			min_z, max_z = min_z_by_intensity, max_z_by_intensity
		self.scan_voxel_array = self.scan_voxel_array[:,:,min_z:max_z]
		self.mask_voxel_array = self.mask_voxel_array[:,:,min_z:max_z]
		self.nb_slices = self.scan_voxel_array.shape[2]

	def crop_z(self, factor, margin):
	    """ Get the z range to keep by calculating mean and std of intensity values of each slice. """
	    self.delete_outliers(-600, -150, margin)
	    intensity_min, intensity_max = self.get_intensity_range(factor) 
	    self.delete_outliers(intensity_min, intensity_max, margin)
	    
	def crop_xy(self):
		""" Reduce slice size by keeping a centered square of size old_side_length/sqrt(2) """
		new_side_length = int(self.side_length/np.sqrt(2))
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
		self.side_length = new_side_length

	def crop_3d(self, factor, margin):
	    self.crop_z(factor, margin)
	    self.crop_xy()


# +-------------------------------------------------------------------------------------+ #
# |                                   		  CUT  		                                | #
# +-------------------------------------------------------------------------------------+ #

	def find_extremal_non_black_slices(self):
		""" find first and last slice with white pixels"""
		min_z, max_z = None, None
		white_pixels_per_slice = np.count_nonzero(self.mask_voxel_array, axis=(0,1))
		for z in range(self.nb_slices):
			if white_pixels_per_slice[z] > 0:
				min_z = z
				break
		for z in range(self.nb_slices-1,-1,-1):
			if white_pixels_per_slice[z] > 0:
					max_z = z
					break
		return min_z, max_z

	def find_padding_params(self, target_depth):
		''' from the patient's mask find the first and last non 
			white slices and the offsest to reach a multiple of target depth.
		'''
		min_z, max_z = self.find_extremal_non_black_slices()
		if (min_z, max_z) == (None, None):
			min_z, max_z = 0, self.nb_slices
		new_z = ((max_z-min_z)//target_depth + 1)*target_depth
		offset = (new_z-(max_z-min_z))//2
		return min_z, max_z, new_z, offset

	def pad_z(self, target_depth):
		""" pad scan and mask to the closest superior multiple of target_depth"""
		min_z, max_z, new_z, offset = self.find_padding_params(target_depth)
		padded_scan = np.zeros((self.side_length, self.side_length, new_z))
		padded_mask = np.zeros((self.side_length, self.side_length, new_z))
		for i in range(max_z-min_z):
			padded_scan[:,:,i+offset] = self.scan_voxel_array[:,:,i+min_z]
			padded_mask[:,:,i+offset] = self.mask_voxel_array[:,:,i+min_z]
		self.scan_voxel_array = padded_scan
		self.mask_voxel_array = padded_mask
		self.nb_slices = new_z

	def cut_z(self, target_depth):
		""" Cut padded scan and mask into several cubes of size target_depth.
			Note that this function will pad scan and mask if the current depth (ie nb_slices)
			isn't a multiple of target_depth.
		"""
		current_depth = self.nb_slices
		assert current_depth % target_depth == 0
		offset = 0
		cutted_scans, cutted_masks = [], []
		while offset <= (current_depth//target_depth -1)*target_depth:
			cutted_scan = np.ndarray((self.side_length, self.side_length, target_depth))
			cutted_mask = np.ndarray((self.side_length, self.side_length, target_depth))
			cutted_scan = self.scan_voxel_array[:,:,offset:offset+target_depth]
			cutted_mask = self.mask_voxel_array[:,:,offset:offset+target_depth]
			cutted_scans.append(cutted_scan)
			cutted_masks.append(cutted_mask)
			offset += target_depth
		return cutted_scans, cutted_masks

	def copy_xy_data(self, input_array, output_shape, x_offset, y_offset):
		output_array = np.ndarray(output_shape)
		for x in range(output_shape[0]):
			for y in range(output_shape[1]):
				output_array[x,y,:] = input_array[x+x_offset,y+y_offset,:]
		return output_array

	def cut_xy_in_four(self, array):
		assert array.shape[0] == array.shape[1]
		half_length = array.shape[0]//2
		new_shape = (half_length, half_length, array.shape[2])
		top_left  = self.copy_xy_data(array, new_shape, 0,  0)
		top_right = self.copy_xy_data(array, new_shape, 0,  half_length)
		bot_left  = self.copy_xy_data(array, new_shape, half_length, 0)
		bot_right = self.copy_xy_data(array, new_shape, half_length, half_length)
		return {'top_left': top_left, 'top_right': top_right, 
				'bot_left': bot_left, 'bot_right': bot_right}

	def cut(self, target_depth):
		""" If self.scan_voxel_array and self.mask_voxel are currently of 
			size (s,s,d), this function will create several cubes 
			of size (s//2,s//2, target_depth), by doing:
			1. z padding
			2. z cutting
			3. for each z cutted cubes, cut each slice in four 
		"""
		self.pad_z(target_depth)
		cutted_scans, cutted_masks = self.cut_z(target_depth)
		for i in range(len(cutted_scans)):
			self.cutted_scans.append(self.cut_xy_in_four(cutted_scans[i]))
			self.cutted_masks.append(self.cut_xy_in_four(cutted_masks[i]))


# +-------------------------------------------------------------------------------------+ #
# |                                   	  AUGMENT  		                                | #
# +-------------------------------------------------------------------------------------+ #

	def augment(self, p, cutted_before=True):
		augmentation = Compose([
							ElasticTransform((0, 0.25)),
							Rotate((-15,15),(-15,15),(-15,15)),
							Flip(0),
							Flip(1),
							Flip(2),
							RandomRotate90((0,1)),
							RandomGamma(),
							GaussianNoise(),
							Normalize(always_apply=True)
						], p=p)
		data = {'image': self.scan_voxel_array, 'mask': self.mask_voxel_array}
		aug_data = augmentation(**data)
		self.augmented_scan, self.augmented_mask = aug_data['image'], aug_data['mask']

# +-------------------------------------------------------------------------------------+ #
# |                                   	   I/O  		                                | #
# +-------------------------------------------------------------------------------------+ #

	def load_scan(self, scan_path):
		self.scan_voxel_array     = np.load(scan_path)
		self.side_length      	  = self.scan_voxel_array.shape[0]
		self.nb_slices		      = self.scan_voxel_array.shape[2]
		self.mean_intensity_stack = self.scan_voxel_array.mean(axis=(0,1))

	def load_mask(self, mask_path):
		self.mask_voxel_array = np.load(mask_path)

	def save_scan(self, scan_path):
		np.save(scan_path, self.scan_voxel_array)

	def save_mask(self, mask_path):
		np.save(mask_path, self.mask_voxel_array)

	def save_cutted_scans(self, name):
		for i in range(len(self.cutted_scans)):
			np.save(f"{name}_[z{i}_top_left]",  self.cutted_scans[i]['top_left'])
			np.save(f"{name}_[z{i}_top_right]", self.cutted_scans[i]['top_right'])
			np.save(f"{name}_[z{i}_bot_left]",  self.cutted_scans[i]['bot_left'])
			np.save(f"{name}_[z{i}_bot_right]", self.cutted_scans[i]['bot_right'])

	def save_cutted_masks(self, name):
		for i in range(len(self.cutted_masks)):
			np.save(f"{name}_[z{i}_top_left]",  self.cutted_masks[i]['top_left'])
			np.save(f"{name}_[z{i}_top_right]", self.cutted_masks[i]['top_right'])
			np.save(f"{name}_[z{i}_bot_left]",  self.cutted_masks[i]['bot_left'])
			np.save(f"{name}_[z{i}_bot_right]", self.cutted_masks[i]['bot_right'])

	def save_augmented_scan(self, name):
		np.save(name, self.augmented_scan)
	
	def save_augmented_mask(self, name):
		np.save(name, self.augmented_mask)
		

		