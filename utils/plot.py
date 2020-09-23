from skimage.measure import label, regionprops

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle




def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image


def compare_3_images(image1, image2, image3):
    f, plots = plt.subplots(1,3, figsize=(50,50))
    plots[0].axis('off')
    plots[0].imshow(image1, cmap=plt.cm.bone)
    plots[1].axis('off')
    plots[1].imshow(image2, cmap=plt.cm.bone)
    plots[2].axis('off')
    plots[2].imshow(image3, cmap=plt.cm.bone)


def compare_4_images(image1, image2, image3, image4):
    f, plots = plt.subplots(2,2, figsize=(50,50))
    plots[0,0].axis('off')
    plots[0,0].imshow(image1, cmap=plt.cm.bone)
    plots[0,1].axis('off')
    plots[0,1].imshow(image2, cmap=plt.cm.bone)
    plots[1,0].axis('off')
    plots[1,0].imshow(image3, cmap=plt.cm.bone)
    plots[1,1].axis('off')
    plots[1,1].imshow(image4, cmap=plt.cm.bone)


def plot_slices(scan, step):
    mean_intensity_stack = scan.mean(axis=(0,1))
    depth = scan.shape[2]
    f, plots = plt.subplots(int(depth / (4*step)) + 1, 4, figsize=(25, 25))
    for i in range(0, int(depth/step)):
        scan_slice = scan[:,:,i*step]
        plots[int(i / 4), int((i % 4))].set_title(str(mean_intensity_stack[i*step]))
        plots[int(i / 4), int((i % 4))].axis('off')
        plots[int(i / 4), int((i % 4))].imshow(scan_slice, cmap=plt.cm.bone)


def big_plot(image, n=10, color_map=plt.cm.bone):
    f, plot = plt.subplots(1, 1, figsize=(n, n))
    plot.axis('off')
    plot.imshow(image, cmap=color_map)


def plot_patched_voxel_array_slice_with_annotations(voxel_array, annotations):
    voxel_array_slice = voxel_array[:,:,annotations[0][2]]
    f, plot = plt.subplots(1, 1, figsize=(10, 10))
    for x,y,z in annotations:
        circle = Circle((x,y), 20, color='r', fill=False)
        plot.add_patch(circle)
    #plot.axis('off')
    plot.imshow(voxel_array_slice, cmap=plt.cm.bone)


def plot_patched_mask_with_annotations(mask, annotations):
    f, plot = plt.subplots(1, 1, figsize=(10, 10))
    for x,y,z in annotations:
        circle = Circle((x,y), 20, color='r', fill=False)
        plot.add_patch(circle)
    #plot.axis('off')
    plot.imshow(mask, cmap=plt.cm.bone)


def plot_bboxes_with_annotation_on_mask(mask, annotation):
    f, plot = plt.subplots(1, 1, figsize=(10, 10))
    label_image = label(mask)
    regions = regionprops(label_image)
    for region in regions:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
        plot.add_patch(rect)
    x, y = int(annotation[0]), int(annotation[1])
    circle = Circle((x,y), 20, color='g', fill=False)
    plot.add_patch(circle)
    plot.imshow(mask, cmap=plt.cm.bone)