import torch
import SimpleITK as sitk
import numpy as np
import cv2
from scipy import ndimage

if __name__ == "__main__":
    volumepath = '/dataT1/Free/qinan/fulllabel/ct/IleusAMU112_in1mm_phase0.nii.gz'
    labelpath = '/dataT1/Free/qinan/fulllabel/label/IleusAMU112_in1mm_phase0.nii.gz'
    ct = sitk.ReadImage(volumepath, sitk.sitkFloat32)
    ct_array = sitk.GetArrayFromImage(ct)
    gt = sitk.ReadImage(labelpath, sitk.sitkUInt8)
    gt_array = sitk.GetArrayFromImage(gt)
    print('ct_array shape = ',ct_array.shape)
    print('gt_array shape = ',gt_array.shape)

    slice1 = ct_array[:,:,125]
    print('slice1 shape = ',slice1.shape)
    cv2.imwrite('slice1.png', slice1)

    gt1 = gt_array[:,:,125]
    print('gt1 shape = ',gt1.shape)
    cv2.imwrite('gt1.png', gt1*255)
    
    slice2 = ct_array[125]
    print('slice2 shape = ',slice2.shape)
    cv2.imwrite('slice2.png', slice2)

    gt2 = gt_array[125]
    ground_map = ndimage.distance_transform_edt(gt2)
    normalized_distance_map = cv2.normalize(ground_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    colored_distance_map = cv2.applyColorMap((normalized_distance_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('gt2 dismap.png',colored_distance_map)
    print('gt2 shape = ',gt2.shape)
    cv2.imwrite('gt2.png', gt2*255)

    slice3 = ct_array[:,125,:]
    print('slice3 shape = ',slice3.shape)
    cv2.imwrite('slice3.png', slice3)

    gt3 = gt_array[:,125,:]
    print('gt3 shape = ',gt3.shape)
    cv2.imwrite('gt3.png', gt3*255)
    
    
    

