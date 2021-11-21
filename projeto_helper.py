# Alunos:
# Luís Felipe Corrêa Ortolan, 759375
# Marco Antônio Bernardi Grivol, 758619

import cv2
import sys
import numpy as np
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
from PIL import Image

class BeerClassification:
    """
    Helper class for identifying outliers in beer labels
    """

    def __init__(
        self, 
        folder_path, 
        ids):
        """Constructor

        Parameters
        ----------
        folder_path : str 
            path to folder containing the folders with images from each id

        ids : list of strings
            each subfolder id inside folder_path
        """
        self.imgs = []
        self.labels = []
        self.ids = ids
        for i in ids:
            img_with_names = self._getImagesFromFolder(join(folder_path, i))
            self.imgs.extend(img_with_names[0])
            self.labels.extend(img_with_names[1])
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)

    def processGetDiffAllImages(
        self,
        query_imgs,
        query_idxs,
        masks,
        folder_path='',
        kernel=(9, 9),
        nfeatures=4000, 
        nOctaveLayers=6, 
        edgeThreshold=4, 
        sigma=2.5):

        """Uses SIFT and RANSAC to obtain a transformation for the train_img
        and returns the difference between both images, does this process for 
        all the loaded images.

        Parameters
        ----------
        query_imgs : ndarray of images
            Target images to be used as reference for the transformation.

        query_idxs : list of integer
            Indexes of the query_imgs.

        masks : ndarray of images
            Masks used with query_imgs

        folder_path : strin, optional
            Folder path to save the image differences (default is same directory).
        
        kernel : tuple, optional
            Kernel used for GaussianBlur (default is (9, 9)).
        
        nfeatures : integer, optional
            Maximum number of features, check opencv documentation (default is 4000).

        nOctaveLayers : integer, optional
            Number of layers in each octave, check opencv documentation (default is 6).

        edgeThreshold : integer, optional
            Threshold to filter out edge-like features, check opencv documentation (default is 4).

        sigma : float, optional
            Sigma for the Gaussian, check opencv documentation (default is 2.5).

        Returns
        -------
        No return;
        """

        # remove query_imgs to avoid subtraction of the same image
        labels = np.delete(self.labels, query_idxs)
        imgs = np.delete(self.imgs, query_idxs)
        for i, label in enumerate(labels):
            ident, version, rot = label.split('_') # [id, version, rot.jpg]
            rot = rot.split('.')[0] # [rot, jpg]

            train_img = plt.imread(imgs[i])
            if rot == '0':
                img_diff = self.processGetDiff(query_imgs[0], masks[0], train_img, False,
                    kernel, nfeatures, nOctaveLayers, edgeThreshold, sigma)
            elif rot == '45L':
                img_diff = self.processGetDiff(query_imgs[1], masks[1], train_img, False,
                    kernel, nfeatures, nOctaveLayers, edgeThreshold, sigma)
            elif rot == '45R':
                img_diff = self.processGetDiff(query_imgs[2], masks[2], train_img, False,
                    kernel, nfeatures, nOctaveLayers, edgeThreshold, sigma)
            else:
                sys.exit(f'Unknow rotation ({rot}) in idx: {i}. Label: {label}')
            
            img_diff_label = f'{ident}_DIFF{version}_{rot}.jpg'
            self._saveImage(img_diff, join(folder_path, img_diff_label))
    
    def processGetDiff(
        self, 
        query_img, 
        query_img_mask,
        train_img, 
        plot=False,
        kernel=(9,9),
        nfeatures=4000, 
        nOctaveLayers=6, 
        edgeThreshold=4, 
        sigma=2.5):
        """Uses SIFT and RANSAC to obtain a transformation for the train_img
        and returns the difference between both images.

        Parameters
        ----------
        query_img : ndarray
            Target image to be used as reference for the transformation.

        query_img_mask : ndarray of images
            Masks used with query_imgs

        train_img : ndarray
            Image to be transformed to match the query image.

        plot : bool, optional
            If True, uses matplotlib to show each step of the process (default
            is False).

        kernel : tuple, optional
            Kernel used for GaussianBlur (default is (9, 9)).
        
        nfeatures : integer, optional
            Maximum number of features, check opencv documentation (default is 4000).

        nOctaveLayers : integer, optional
            Number of layers in each octave, check opencv documentation (default is 6).

        edgeThreshold : integer, optional
            Threshold to filter out edge-like features, check opencv documentation (default is 4).

        sigma : float, optional
            Sigma for the Gaussian, check opencv documentation (default is 2.5).
        
        Returns
        -------
        img_diff : ndarray
            Bitwise difference of query and train images. 
        """
        ret = self.transform(query_img, train_img, kernel, nfeatures, nOctaveLayers, edgeThreshold, sigma)
        train_img_t = ret[0]
        train_img_t = cv2.bitwise_and(train_img_t, query_img_mask)

        query_img_s = cv2.GaussianBlur(query_img, kernel, cv2.BORDER_DEFAULT)
        train_img_s = cv2.GaussianBlur(train_img_t, kernel, cv2.BORDER_DEFAULT)
        
        img_diff = cv2.subtract(train_img_s, query_img_s)
        img_diff_crop = self.crop(img_diff)
        
        if plot:
            query_kpts = ret[1]
            train_kpts = ret[2]
            matches = ret[3]
            self._plot_imgs(query_img, train_img, query_kpts, train_kpts,
                            matches, train_img_t, img_diff_crop)
        return img_diff_crop
    
    def transform(
        self, 
        query_img, 
        train_img,
        kernel=(9, 9),
        nfeatures=4000, 
        nOctaveLayers=6, 
        edgeThreshold=4, 
        sigma=2.5):
        """Uses SIFT and RANSAC to obtain a Homography matrix to match train_img
        label with query_img.

        Parameters
        ----------
        query_img : ndarray
            Target image to be used as reference for the transformation.

        train_img : ndarray
            Image to be transformed to match the query image.

        kernel : tuple, optional
            Kernel used for GaussianBlur (default is (9, 9)).
        
        nfeatures : integer, optional
            Maximum number of features, check opencv documentation (default is 4000).

        nOctaveLayers : integer, optional
            Number of layers in each octave, check opencv documentation (default is 6).

        edgeThreshold : integer, optional
            Threshold to filter out edge-like features, check opencv documentation (default is 4).

        sigma : float, optional
            Sigma for the Gaussian, check opencv documentation (default is 2.5).

        Returns
        -------
        (train_img_t, query_kpts, train_kpts, matches) : tuple
            train_img_t : ndarray
                Transformed train_img.
            
            query_kpts :  list of cv2.Keypoint
                query_img keypoints.
            
            train_kpts : list of cv2.Keypoint
                train_img keypoints.
            
            matches : list of cv2.matches
                Keypoint matches.
        """
        if kernel == None:
            query_img_s = query_img
            train_img_s = train_img
        else:
            query_img_s = cv2.GaussianBlur(query_img, kernel, cv2.BORDER_DEFAULT)
            train_img_s = cv2.GaussianBlur(train_img, kernel, cv2.BORDER_DEFAULT)

        sift = cv2.SIFT_create(
            nfeatures=nfeatures, 
            nOctaveLayers=nOctaveLayers, 
            edgeThreshold=edgeThreshold, 
            sigma=sigma)
        query_kpts, query_desc = sift.detectAndCompute(query_img_s, None)
        train_kpts, train_desc = sift.detectAndCompute(train_img_s, None)
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(train_desc, query_desc)
        matches = sorted(matches, key=lambda x:x.distance)
        
        H = self.getHomography(query_kpts, train_kpts, matches)

        width = query_img.shape[1]
        height = query_img.shape[0]
        train_img_t = cv2.warpPerspective(train_img, H, (width, height))
        return (train_img_t, query_kpts, train_kpts, matches)
    
    def thresholdAllImages(
        self,
        model,
        train_folder,
        test_folder,
        test_ids=None,
        T=50,
        verbose=True
    ):
        """Apply a binary threshold to all images, sums the white pixel count of
        each image. Uses the train_folder to fit the model and test_folder to evaluate.

        Parameters
        ----------
        model : sklearn object
            sklearn classification model. OneClassSVM and IsolationForest are 
            recommended, but could also use some other as long as it classifies
            outliers with -1 and inliers with 1 and has the "fit" and "predict"
            methods.
        
        train_folder : string
            Path to the folder containing the image differences used for training 
            the model. Should/Must not contain outliers.

        test_folder : string
            Path to the folder containing the image differences used for testing
            the model. May contain outliers.

        test_ids : list of strings, optional
            Image classes used for testing (default is None, meaning all the images
            inside the test folder)

        threshold_value : int, optional
            Pixel values below this threshold will be black and white otherwise
            (default is 50).

        verbose : bool, optional
            Print accuracy and error from the train and test stages (default is
            True).

        Returns
        -------
        train_pred, test_pred : (list, list)
            Returns the train and test predictions respectively.
            1 for inliers
            -1 for outliers
        """

        train_imgs, train_labels = self._getImagesFromFolder(train_folder)
        true_train_labels = []
        train_t_sum = []
        for i, img in enumerate(train_imgs):
            img = plt.imread(img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, img_t = cv2.threshold(img_gray, T, 1, cv2.THRESH_BINARY)
            train_t_sum.append(np.sum(img_t))

            ident = train_labels[i].split('_')[0] # id_version_rot.jpg
            if ident == '0':
                true_train_labels.append(1)
            else:
                true_train_labels.append(-1)

        max_t = np.max([np.max(train_t_sum), 1])
        train_t_sum = train_t_sum / max_t

        test_imgs, test_labels = self._getImagesFromFolder(test_folder)
        true_test_labels = []
        test_t_sum = []
        for i, img in enumerate(test_imgs):
            ident = test_labels[i].split('_')[0] # id_version_rot.jpg
            if test_ids != None and ident not in test_ids:
                continue
            img = plt.imread(img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, img_t = cv2.threshold(img_gray, T, 1, cv2.THRESH_BINARY)
            test_t_sum.append(np.sum(img_t))

            if ident == '0':
                true_test_labels.append(1)
            else:
                true_test_labels.append(-1)

        max_t = np.max([np.max(test_t_sum), 1])
        test_t_sum = test_t_sum / max_t

        train_t_sum = np.array(train_t_sum).reshape((len(train_t_sum), 1))
        test_t_sum = np.array(test_t_sum).reshape((len(test_t_sum), 1))

        model.fit(train_t_sum)

        train_pred = model.predict(train_t_sum)
        train_error = np.sum(train_pred != true_train_labels)
        train_acc = 1.0 - train_error / len(train_t_sum)

        test_pred = model.predict(test_t_sum)
        test_error = np.sum(test_pred != true_test_labels)
        test_acc = 1.0 - test_error / len(test_t_sum)

        if verbose:
            print('Train ->', end='')
            print(f'    Error: {train_error}/{len(train_t_sum)} | {train_error / len(train_t_sum):.3f}', end='')
            print(f'    Acuracy: {train_acc:.3f}')
            print('Test -> ', end='')
            print(f'    Error: {test_error}/{len(test_t_sum)} | {test_error / len(test_t_sum):.3f}', end='')
            print(f'    Acuracy: {test_acc:.3f}')
        return train_pred, test_pred

    def compareHistogramAllImages(
        self,
        model,
        train_folder,
        test_folder,
        test_ids=None,
        bins=[10],
        ranges=[10, 100],
        verbose=True
    ):
        """Calculates the histogram of the difference image. Uses the train_folder 
        to fit the model and test_folder to evaluate.

        Parameters
        ----------
        model : sklearn object
            sklearn classification model. OneClassSVM and IsolationForest are 
            recommended, but could also use some other as long as it classifies
            outliers with -1 and inliers with 1 and has the "fit" and "predict"
            methods.
        
        train_folder : string
            Path to the folder containing the image differences used for training 
            the model. Should/Must not contain outliers.

        test_folder : string
            Path to the folder containing the image differences used for testing
            the model. May contain outliers.

        test_ids : list of strings, optional
            Image classes used for testing (default is None, meaning all the images
            inside the test folder)

        bins : [integer]
            Bins to divide the histogram.

        ranges : [min, max]
            Ranges to consider for the histogram.

        verbose : bool, optional
            Print accuracy and error from the train and test stages (default is
            True).

        Returns
        -------
        train_pred, test_pred : (list, list)
            Returns the train and test predictions respectively.
            1 for inliers
            -1 for outliers
        """

        train_imgs, train_labels = self._getImagesFromFolder(train_folder)
        train_histograms = []
        true_train_labels = []
        for i, img in enumerate(train_imgs):
            img = plt.imread(img)
            hist = []
            for channel in range(3):
                h = cv2.calcHist([img], [channel], None, bins, ranges)
                h = cv2.normalize(h, h).flatten()
                hist.extend(h)
            train_histograms.append(hist)

            ident = train_labels[i].split('_')[0] # id_version_rot.jpg
            if ident == '0':
                true_train_labels.append(1) # inlier
            else:
                true_train_labels.append(-1) # outlier

        test_imgs, test_labels = self._getImagesFromFolder(test_folder)
        test_histograms = []
        true_test_labels = []
        for i, img in enumerate(test_imgs):
            ident = test_labels[i].split('_')[0] # id_version_rot.jpg
            if test_ids != None and ident not in test_ids:
                continue
            img = plt.imread(img)
            hist = []
            for channel in range(3):
                h = cv2.calcHist([img], [channel], None, bins, ranges)
                h = cv2.normalize(h, h).flatten()
                hist.extend(h)
            test_histograms.append(hist)

            if ident == '0':
                true_test_labels.append(1) # inlier
            else:
                true_test_labels.append(-1) # outlier

        model.fit(train_histograms)

        train_pred = model.predict(train_histograms)
        train_error = np.sum(train_pred != true_train_labels)
        train_acc = 1.0 - train_error / len(train_histograms)

        test_pred = model.predict(test_histograms)
        test_error = np.sum(test_pred != true_test_labels)
        test_acc = 1.0 - test_error / len(test_histograms)

        if verbose:
            print('Train ->', end='')
            print(f'    Error: {train_error}/{len(train_histograms)} | {train_error / len(train_histograms):.3f}', end='')
            print(f'    Acuracy: {train_acc:.3f}')
            print('Test -> ', end='')
            print(f'    Error: {test_error}/{len(test_histograms)} | {test_error / len(test_histograms):.3f}', end='')
            print(f'    Acuracy: {test_acc:.3f}')
        return train_pred, test_pred
    
    def getHomography(self, query_kpts, train_kpts, matches):
        """Returns a homography matrix to transform the train image to match
        the query. 

        Adapted from: https://colab.research.google.com/drive/11Md7HWh2ZV6_g3iCYSUw76VNr4HzxcX5

        Parameters
        ----------
        query_kpts : ndarray of cv2.Keypoint
            Query image keypoints.

        train_kpts : ndarray of cv2.Keypoint
            Train image keypoints.

        matches : cv2.matches
            Matched descriptors of both images, use BeerClassification.matcher.

        Returns
        -------
        H : ndarray
            Homography matrix
        """
        # convert the keypoints to numpy arrays
        train_kpts = np.float32([kp.pt for kp in train_kpts])
        query_kpts = np.float32([kp.pt for kp in query_kpts])

        if len(matches) > 4:
            # construct the two sets of points
            train_pts = np.float32([train_kpts[m.queryIdx] for m in matches])
            query_pts = np.float32([query_kpts[m.trainIdx] for m in matches])

            # estimate the homography between the sets of points
            (H, status) = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 3)
            return H
        else:
            return None

    def crop(self, img):
        """Crops the image to remove rows and columns filled with zeros.

        Parameters
        ----------
        img : ndarray
            Image with black borders

        Returns
        -------
        img_crop : ndarray
            Cropped image
        """
        # finds the first index filled with zeros
        row, col, d = img.shape
        # left
        for i in range(col):
            if np.sum(img[:, i, :]) > 0:
                break
        # right
        for j in range(col - 1, 0, -1):
            if np.sum(img[:, j, :]) > 0:
                break
        # up
        for m in range(row):
            if np.sum(img[m, :, :]) > 0:
                break
        # down
        for n in range(row - 1, 0, -1):
            if np.sum(img[n, :, :]) > 0:
                break
        img_crop = img[m : n + 1, i : j + 1, :]
        return img_crop
    
    def _getImagesFromFolder(self, folder_path):
        """Should not be called from outside method.
        Get all the images inside the folder.

        Parameters
        ----------
        folder_path : string
            Path to the folder with images.

        Returns
        -------
        (imgs, labels) : tuple
            imgs : list of strings
                Path to the image (not the image), use plt.imread(imgs[i]) to read.
            labels : list of strins
                Name of each image in the folder.
        """
        imgs = []
        labels = []
        for file in listdir(folder_path):
            img_path = join(folder_path, file)
            if isfile(img_path) and file.endswith('.jpg'):
                imgs.append(img_path)
                labels.append(file)
        return (imgs, labels)
    
    def _plot_imgs(self, query_img, train_img, query_kpts, train_kpts,
                   matches, train_img_t, img_diff):
        """Should not be called from outside method.
        Plot the process of image feature detection, matching and image difference
        with SIFT and RANSAC.

        Parameters
        ----------
        query_img : ndarray
            Reference image.

        train_img : ndarray
            Image to be transformed to match the query image.

        query_kpts : opencv Keypoints
            Query image keypoints.

        train_kpts : opencv Keypoints
            Train image keypoints.

        matches : opencv Matches
            Keypoint matches.

        train_img_t : ndarray
            Train image transformed.

        img_diff : ndarray
            Image obtained from query and train images subtraction.

        Returns
        -------
        No return.
        """
        fig, axs = plt.subplots(5, 2, figsize=(20, 40), constrained_layout=True)
        # draw original images
        axs[0, 0].imshow(query_img)
        axs[0, 0].set_title('Query img')
        axs[0, 1].imshow(train_img)
        axs[0, 1].set_title('Train img')
        # draw keypoints
        axs[1, 0].imshow(cv2.drawKeypoints(query_img, query_kpts,
                                           None, color=(0, 255, 0)))
        axs[1, 0].set_title(f'Query img Keypoints {len(query_kpts)}')
        axs[1, 1].imshow(cv2.drawKeypoints(train_img, train_kpts, 
                                           None, color=(0, 255, 0)))
        axs[1, 1].set_title(f'Train img Keypoints {len(train_kpts)}')
        # draw matches
        gs = axs[2, 0].get_gridspec()
        axs[2, 0].remove()
        axs[2, 1].remove()
        ax_matches = fig.add_subplot(gs[2, :])
        img_match = cv2.drawMatches(train_img, train_kpts,
                            query_img, query_kpts,
                            matches[:100], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        ax_matches.imshow(img_match)
        ax_matches.set_title('Matches')
        # draw transformed image
        axs[3, 0].imshow(train_img)
        axs[3, 0].set_title('Train img')
        axs[3, 1].imshow(train_img_t)
        axs[3, 1].set_title('Train img transformed')
        # draw image difference
        axs[4, 0].imshow(img_diff)
        axs[4, 0].set_title('Image Difference')
        axs[4, 1].imshow(cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY), 'gray')
        axs[4, 1].set_title('Image DIfference Gray')
        plt.show()
        
    def _saveImage(self, img, path='img.jpg'):
        """Should not be called from outside method.
        Saves the image.

        Parameters
        ----------
        img : ndarray
            Image to be saved.

        path : string, optional
            Path to save the image.

        Returns
        -------
        No return
        """
        img = Image.fromarray(img)
        img.save(path)