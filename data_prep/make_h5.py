import click 
import pandas as pd
import numpy as np
import nibabel as nib
import h5py
from tqdm import tqdm
import random
import os

import numpy as np

def crop_scan(scan, seg, nh=128, nw=128):
    """
    Crop a 4D scan (slices, frames, height, width) around the center of a 3D segmentation mask.

    Args:
        scan (np.ndarray): 4D array (slices, frames, H, W)
        seg (np.ndarray): 3D array (H, W, slices)
        nh (int): Crop height
        nw (int): Crop width

    Returns:
        tscan (np.ndarray): Cropped 4D scan (slices, frames, nh, nw)
    """
    s, f, h, w = scan.shape
    h_seg, w_seg, s_seg = seg.shape

    center_slice_idx = s_seg // 2
    seg_slice = seg[:, :, center_slice_idx]
    coords = np.argwhere(seg_slice == 1)

    # If middle slice has no labels, search others
    if coords.size == 0:
        found = False
        for i in range(s_seg):
            if i == center_slice_idx:
                continue  # Already checked
            seg_slice = seg[:, :, i]
            coords = np.argwhere(seg_slice == 1)
            if coords.size > 0:
                found = True
                break
        if not found:
            # Fallback to image center
            y_center = h // 2
            x_center = w // 2
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            y_center = (y_min + y_max) // 2
            x_center = (x_min + x_max) // 2
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        y_center = (y_min + y_max) // 2
        x_center = (x_min + x_max) // 2

    # Pad scan if needed
    pad_top = max(nh // 2 - y_center, 0)
    pad_bottom = max((y_center + nh // 2) - h, 0)
    pad_left = max(nw // 2 - x_center, 0)
    pad_right = max((x_center + nw // 2) - w, 0)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        scan = np.pad(
            scan,
            pad_width=((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )
        y_center += pad_top
        x_center += pad_left
        h += pad_top + pad_bottom
        w += pad_left + pad_right

    # Compute crop coordinates
    y_start = y_center - nh // 2
    x_start = x_center - nw // 2
    y_start = max(min(y_start, h - nh), 0)
    x_start = max(min(x_start, w - nw), 0)
    y_end = y_start + nh
    x_end = x_start + nw

    # Crop the scan
    tscan = scan[:, :, y_start:y_end, x_start:x_end]
    return tscan

def clip_sampling(scan, type, num_slices=11):
    '''Function to perform clip sampling of the scans, with the output size being HxWx11x10
    Inputs:
        - scan: 4D volume of the cardiac MR of size H x W x num_slices x num_of_frames 
        - type: original (our intended sampling, from 50 frames to 10, stride = 5) or augmented (random sampling) 
    Outputs:
        - clippes_scan: 4D valume clipped to the size H x W x 11 x 10
    '''
    h, w, s, t = scan.shape
    assert type == 'original' or type=='random', "input correct clipping type: 'random' or 'original'"
    clipped_scan = np.zeros((h, w, num_slices, 10))
    num_start = 0
    if s > num_slices:
        if s == 10 or s == 11:
            slices_stride = 2
        else:
            slices_stride = 1
    else:
        slices_stride = 1
    if s < num_slices:
        num_slices = s
    selected_slices = [i for i in range(0, s, slices_stride) if i < num_slices]
    if type == 'original':
        j = 0
        if t == 50:
            for i in range(0, 49, 5):
                if j < 10:
                    clipped_scan[:,:,:len(selected_slices),j] = scan[:,:,selected_slices,i]
                    j+=1
        else:
            stride = t // 10
            j = 0
            for i in range(0, t, stride):
                if j < 10:
                    clipped_scan[:,:,:len(selected_slices),j] = scan[:,:,selected_slices,i]
                    j+=1
    elif type == 'random':
        start = random.randint(0, 40)
        max_stride = (50 - start) // 10
        if max_stride < 5:
            stride = random.randint(1, (50 - start) // 10)
        else:
            stride = random.randint(1, 4)
        j = 0
        for i in range(start, 49, stride):
            if j < 10:
                clipped_scan[:,:,:len(selected_slices),j] = scan[:,:,selected_slices,i]
                j+=1
    return clipped_scan

@click.command()
@click.option('--csv_path', '-p', help='Path to the csv file you want to convert to anndata file', required=True)
@click.option('--output_path', '-o', help='Path to the csv file you want to convert to anndata file', required=True)

def main(csv_path, output_path):
    print("Converting file: ", csv_path)
    root='/lustre/groups/shared/ukbb-87065/dataset/cardiac_mri_nifti'
    df = pd.read_csv(csv_path)
    folder_paths = df["folder_y"].to_numpy()
    eids = df["eid"].to_numpy()

    metadata = df.drop(columns=['I20', 'I21', 'I24', 'I25', 'mr_date'])
    print(metadata.columns)
    with h5py.File(output_path, 'w') as f:
        # Store metadata as a dataset and store column names as an attribute
        metadata_data = metadata.to_numpy(dtype=np.float32)
        metadata_dset = f.create_dataset("metadata", data=metadata_data, compression="gzip", compression_opts=4)
        metadata_dset.attrs["columns"] = list(metadata.columns)  # Store column names as an attribute

        # Create a dataset for images, applying compression and chunking for efficiency
        img_shape = (11, 10, 128, 128)  # 4D shape for your images (time, channels, height, width)
        image_dset = f.create_dataset(
            "images", shape=(0, *img_shape), maxshape=(None, *img_shape),
            dtype=np.float16, chunks=(1, *img_shape), compression="gzip", compression_opts=4
        )

        # Read NIfTI images, convert to numpy arrays, and write them into the HDF5 dataset
        for i, path in enumerate(tqdm(folder_paths)):
            eid = eids[i]
            if len(str(path)) == 1:
                path = "0" + str(path)
            else:
                path = str(path)
            path = os.path.join(root, path)
            path = os.path.join(path, str(eid))
            
            # Load the NIfTI image
            nifti_img = nib.load(os.path.join(path, 'sa.nii.gz'))
            nifti_seg = nib.load(os.path.join(path, 'seg_sa_ED.nii.gz')).get_fdata()
            img_data = nifti_img.get_fdata()  # Convert to NumPy array
            
            # Preprocessing (if any)
            img_data = clip_sampling(img_data, 'original').transpose(2, 3, 0, 1)  # Assuming clip_sampling adjusts shape
            img_data = crop_scan(img_data, nifti_seg)  # Assuming this step pads or crops the images
            
            # Write image data to the HDF5 dataset (append to it)
            image_dset.resize(image_dset.shape[0] + 1, axis=0)  # Resize to append
            image_dset[-1] = img_data.astype(np.float16)  # Store the image as float16

        print(f"HDF5 file saved as {output_path}")

    
if __name__ == '__main__':
    main()   