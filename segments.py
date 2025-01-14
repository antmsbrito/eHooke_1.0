﻿"""Module responsible for the identification of single regions inside the
mask, which should correspond to the cell regions.
The regions are then labeled using the watershed algorithm. Requires an
ImageManager object containg the loaded images (base + fluor) and the mask.
Contains a single class, Segments, which stores the data from the processing
of the mask: features and labels, which will later be used to define the different cell regions
"""

import numpy as np
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage
from skimage.io import imsave
from skimage.util import img_as_int
from tkinter.filedialog import asksaveasfilename

# AB
from skimage.transform import EuclideanTransform, warp


class SegmentsManager(object):
    """Main class of the module.
    The class is responsible for the computation of the features of the image.
    Requires an instance of the ImageManager class after loading both the
    base and fluor and image aswell as computing the mask
    The first step of the analysis consists on the usage of the distance
    peaks algorithm (based on the Euclidean Distances) and the second on the
    usage of the watershed algorithm to define each individual region of the
    mask"""

    def __init__(self):
        self.features = None
        self.labels = None
        self.base_w_features = None
        self.fluor_w_features = None

    def clear_all(self):
        """Resets the class instance to the initial state"""
        self.features = None
        self.labels = None
        self.base_w_features = None
        self.fluor_w_features = None

    @staticmethod
    def compute_distance_peaks(mask, params):
        """Method used when the selected algorithm for the feature computation
        is the Distance Peaks.
        Returns a list of the centers of the different identified regions,
        which should be used in the compute_features method"""

        distance = ndimage.morphology.distance_transform_edt(1 - mask)

        mindist = params.peak_min_distance
        minmargin = params.peak_min_distance_from_edge

        centers = peak_local_max(distance, min_distance=mindist,
                                 threshold_abs=params.peak_min_height,
                                 exclude_border=True,
                                 num_peaks=params.max_peaks,
                                 indices=True)

        placedmask = np.ones(distance.shape)
        lx, ly = distance.shape
        result = []
        heights = []
        circles = []

        for c in centers:
            x, y = c

            if x >= minmargin and y >= minmargin and x <= lx - minmargin \
                    and y <= ly - minmargin and placedmask[x, y]:
                placedmask[x - mindist:x + mindist +
                                       1, y - mindist:y + mindist + 1] = 0
                s = distance[x, y]
                circles.append((x, y))
                heights.append(s)

        ixs = np.argsort(heights)
        for ix in ixs:
            result.append(circles[ix])

        return result

    def compute_features(self, params, image_manager):
        """Method used to compute the features of an image using the mask.
        requires a mask and an instance of the imageprocessingparams
        if the selected algorithm used is Distance Peak, used the method
        compute_distance_peaks to compute the features"""

        mask = image_manager.mask
        features = np.zeros(mask.shape)

        if params.peak_min_distance_from_edge < 1:
            params.peak_min_distance_from_edge = 1

        if image_manager.stardist_polygons is not None:
            circles = image_manager.stardist_polygons['points']
        else:
            circles = self.compute_distance_peaks(mask, params)

        for ix, c in enumerate(circles):
            x, y = c
            for f in range(3):
                features[x - 1 + f, y] = ix + 1
                features[x, y - 1 + f] = ix + 1

        self.features = features

    def overlay_base_w_features(self, image_manager):
        """Method used to produce an image with an overlay of the features on
        the base image requires a base image, the features and the clip
        values to overlay the images returns a image matrix which can be saved
        using the save_image method from EHooke or directly using the imsave
        from skimage.io"""

        clipped_base = np.copy(image_manager.base_image)

        places = self.features > 0.5
        clipped_base[places] = 1
        self.base_w_features = clipped_base

    def overlay_fluor_w_features(self, image_manager):
        """Method used to produce an image with an overlay of the features on
        the base image requires a base image, the features and the clip
        values to overlay the imagesreturns a image matrix which can be saved
        using the save_image method from EHooke or directly using the imsave
        from skimage.io"""

        clipped_fluor = np.copy(image_manager.fluor_image)

        places = self.features > 0.5
        clipped_fluor[places] = 1
        self.fluor_w_features = clipped_fluor

    def compute_labels(self, params, image_manager):
        """Computes the labels for each region based on the previous computed
        features. Requires the mask, th base mask, the features and an
        instance of the imageprocessingparams"""

        if image_manager.stardist_labels is not None:
            self.labels = image_manager.stardist_labels
        else:
            markers = self.features
            base_mask = image_manager.base_mask
            mask = image_manager.mask
            inverted_mask = 1 - image_manager.mask

            if params.outline_use_base_mask:
                tmpmask = np.ones(mask.shape)
                tmpmask[base_mask > mask] = 0
                distance = ndimage.morphology.distance_transform_edt(tmpmask)

            else:
                distance = - \
                    ndimage.morphology.distance_transform_edt(inverted_mask)

            mindist = np.min(distance)
            markpoints = markers > 0
            distance[markpoints] = mindist
            labels = morphology.watershed(distance, markers, mask=inverted_mask)

            self.labels = labels

    def compute_segments(self, params, image_manager):
        """Calls the different methods of the module in the right order.
        Can be used as the interface of this module in the main module of the
        software"""

        self.compute_features(params, image_manager)
        self.overlay_base_w_features(image_manager)
        self.overlay_fluor_w_features(image_manager)
        self.compute_labels(params, image_manager)

    def save_labels(self, filename=None):
        if filename is None:
            filename = asksaveasfilename()

        imsave(filename + ".tif", self.labels)

    def save_fluor(self, filename, image_manager):
        imsave(filename + ".tif", image_manager.original_fluor_image)

    def save_labels_aligned(self, filename, image_manager):
        best = image_manager.align_values
        dy, dx = best
        final_matrix = EuclideanTransform(rotation=0, translation=(-dx, -dy))
        mask_aligned = warp(self.labels, final_matrix.inverse, preserve_range=True)

        # ONLY DO THIS IN CASES WHERE THE VALUES HAVE NO PHYSICAL SIGNIFICANCE
        mask_aligned = np.array(mask_aligned, dtype=np.int32)
        # CARE

        imsave(filename + '.tif', mask_aligned)
