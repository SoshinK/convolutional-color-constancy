import numpy as np
import tqdm
import cv2 as cv2
from minimizer import fit_filter
from scipy.signal import convolve2d
from dataset import Dataset, AugList, LogChromHist, Resize

class GaussianPyramid:
    def __init__(self, num_levels, filter_size=5):
        self.num_levels = num_levels
        self.filter_size = filter_size
        self.filters = None
    
    def fit(self, dataset: Dataset, lambda_regularization=0.1):
        if isinstance(dataset.augmentations, AugList):
            if not isinstance(dataset.augmentations.augmentations_list[-1], LogChromHist):
                raise TypeError("Last transform in dataset should be LogChromHist")
            u_ticks, v_ticks = dataset.augmentations.augmentations_list[-1].histogram.u_coords, dataset.augmentations.augmentations_list[-1].histogram.v_coords
        elif not isinstance(dataset.augmentations, LogChromHist):
            raise TypeError("Last transform in dataset should be LogChromHist")
        else:
            u_ticks, v_ticks = dataset.augmentations.histogram.u_coords, dataset.augmentations.augmentations_list[-1].histogram.v_coords

        orig_transforms = dataset.augmentations
        orig_img_size = dataset[0][0].shape
        dataset = dataset.copy()
        
        self.filters = []
        
        for i in tqdm.tqdm(range(self.num_levels)):
            dataset.augmentations = AugList([orig_transforms, Resize((orig_img_size[0] // 2 ** i, orig_img_size[1] // 2 ** i))])
            train_imgs, data_gt = dataset.get_data()
            self.filters.append(fit_filter(train_imgs, data_gt, lambda_regularization, self.filter_size, u_ticks, v_ticks).reshape((self.filter_size, self.filter_size)))
        print("Pyramid fitted")

    def apply(self, img):
        if self.filters is None:
            raise RuntimeError("Filters haven't been fitted yet")
        result = 0
        pyramid = []
        for i in range(self.num_levels):
            pyramid.append(cv2.resize(img, (img.shape[0] // 2 ** i, img.shape[1] // 2 ** i), interpolation=cv2.INTER_LINEAR))
        result = 0
        for i in range(self.num_levels - 1, 0, -1):
            result += convolve2d(pyramid[i], self.filters[i], mode='same', boundary='fill', fillvalue=0)
            result = cv2.resize(result, (result.shape[0] * 2, result.shape[1] * 2), interpolation=cv2.INTER_LINEAR)
            result = cv2.GaussianBlur(result,(3,3),0)
        result += convolve2d(pyramid[0], self.filters[0], mode='same', boundary='fill', fillvalue=0)
        return result 

