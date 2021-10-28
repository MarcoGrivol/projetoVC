import cv2
import numpy as np
from os import listdir
from os.path import join, isfile
from collections import OrderedDict
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image

class BeerClassification:
    """
    Helper class for identifying problem in beer labels
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
        for i in ids:
            img_with_names = self._getImagesFromFolder(join(folder_path, i))
            self.imgs.extend(img_with_names[0])
            self.labels.extend(img_with_names[1])
    
    def processGetDiff(
        self, 
        query_img, 
        train_img, 
        plot=False):
        """Uses SIFT and RANSAC to obtain a transformation for the train_img
        and returns the difference between both images.

        Parameters
        ----------
        query_img : ndarray
            Target image to be used as reference for the transformation.

        train_img : ndarray
            Image to be transformed to match the query image.

        plot : bool, optional
            If True, uses matplotlib to show each step of the process (default
            is False).
        
        Returns
        -------
        img_diff : ndarray
            Bitwise difference of query and train images. 
        """
        ret = self.transform(query_img, train_img)
        train_img_t = ret[0]
        img_diff = self.imgDiff(query_img, train_img_t)
        
        if plot:
            query_kpts = ret[1]
            train_kpts = ret[2]
            matches = ret[3]
            self._plot_imgs(query_img, train_img, query_kpts, train_kpts,
                            matches, train_img_t, img_diff)
        return img_diff
    
    def transform(self, query_img, train_img):
        """Uses SIFT and RANSAC to obtain a Homography matrix to match train_img
        label with query_img.

        Parameters
        ----------
        query_img : ndarray
            Target image to be used as reference for the transformation.

        train_img : ndarray
            Image to be transformed to match the query image.

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
        query_kpts, query_desc = self.sift(query_img)
        train_kpts, train_desc = self.sift(train_img)
        
        matches = self.matcher(query_desc, train_desc)
        
        H = self.getHomography(query_kpts, train_kpts, matches)
        train_img_t = self.warpPerspective(query_img, train_img, H)
        return (train_img_t, query_kpts, train_kpts, matches)
    
    def thresholdAllImages(self, query_img, query_idx, threshold_value=50):
        """Apply a binary threshold to all images using query_img as reference and
        returns the sum of white pixels from each image.

        Parameters
        ----------
        query_img : ndarray
            Image to be used as reference in the transformation operations.
        
        query_idx : int
            The index of the query_img.

        threshold_value : int, optional
            Pixel values below this threshold will be black and white otherwise
            (default is 50).

        Returns
        -------
        results : (N, 2) list
            Each row contains the index of the image in self.imgs and the sum
            of white pixels after the threshold operation.

        """
        query_label = self.labels[query_idx]
        query_rot = query_label.split('_')[2].split('.')[0]
        
        results = []
        for i, label in enumerate(self.labels):
            rot = label.split('_')[2].split('.')[0] # [id, version, rot.jpg] -> [rot, jpg]
            if rot == query_rot and i != query_idx:
                # rotacao correta e imagem é diferente
                img_diff = self.processGetDiff(query_img, self.getImage(self.imgs[i]))
                t = self.thresholdSum(img_diff, threshold_value)
                results.append([i, t])
        return np.array(results)

    def histCorr(self, query_img, query_idx):
        query_label = self.labels[query_idx]
        query_rot = query_label.split('_')[2].split('.')[0]
        
        color = ('r', 'g', 'b')
        results = []
        for i, label in enumerate(self.labels):
            rot = label.split('_')[2].split('.')[0] # [id, version, rot.jpg] -> [rot, jpg]
            if rot == query_rot and i != query_idx:
                # rotacao correta e imagem é diferente
                img_diff = self.processGetDiff(query_img, self.getImage(self.imgs[i]))
                img_diff_crop = self.trim(img_diff)
                hist_corr = []
                for j, col in enumerate(color):
                    hist = cv2.calcHist([img_diff], [j], None, [256], [0, 255])
                results.append(hist)
        return np.array(results)
    
    def ssimAllImages(self, query_img, query_idx, threshold_value=50):
        query_label = self.labels[query_idx]
        query_rot = query_label.split('_')[2].split('.')[0]
        
        results = []
        for i, label in enumerate(self.labels):
            rot = label.split('_')[2].split('.')[0]
            if rot == query_rot and i != query_idx:
                # rotacao correta e imagem é diferente
                img_diff = self.processGetDiff(query_img, self.getImage(self.imgs[i]))
                
                img_diff_gray = cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY)
                query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
                _, img_t = cv2.threshold(img_diff_gray, threshold_value, 1, cv2.THRESH_BINARY)
                
                ssim_value = ssim(query_img_gray, img_t, data_range=img_t.max() - img_t.min())
                results.append([i, ssim_value])
        return np.array(results)
    
    def thresholdSum(self, img, threshold_value):
        """Convert to gray, apply a binary threshold operation and sum the white
        pixels.

        Parameters
        ----------
        img : ndarray
            RGB image.

        threshold_value : int
            Value for the binary threshold operation.

        Returns
        -------
            np.sum(img_t) : int
                Sum of white pixels.
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t, img_t = cv2.threshold(img_gray, threshold_value, 1, cv2.THRESH_BINARY)
        return np.sum(img_t)
        
    def sift(self, img):
        """Applies SIFT to img.

        Parameters
        ----------
        img : ndarray
            Image.

        Returns:
            (kpts, desc) : tuple
                kpts : cv2.Keypoint
                    Keypoints of the image.
                
                desc : ndarray
                    Descriptos of the image.
        """
        s = cv2.SIFT_create()
        kpts, desc = s.detectAndCompute(img, None)
        return (kpts, desc)
    
    def matcher(self, query_desc, train_desc):
        """Uses a brute force method to obtain the matches from the query and 
        train images descriptos.

        Parameters
        ----------
        query_desc : ndarray
            Query image descriptors.
        
        train_desc : ndarray
            Train image descriptors.

        Returns
        -------
        matches : cv2.matches
            Matched descriptors.
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(train_desc, query_desc)
        matches = sorted(matches, key=lambda x:x.distance)
        return matches
    
    def getHomography(self, query_kpts, train_kpts, matches):
        """Returns a homography matrix to transform the train image to match
        the query. 

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
            (H, status) = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 4)
            return H
        else:
            return None
        
    def warpPerspective(self, query_img, train_img, H):
        """Applies a transformation to train_img using the matrix H.

        Parameters
        ----------
        query_img : ndarray
            Query image used as reference for the transformation.

        train_img : ndarray
            Train image to be transformed.

        H : ndarray
            Transformation matrix.

        Returns
        -------
        train_img_transformed : ndarray
            Train image after geometrical transformation with H.
        """
        width = query_img.shape[1]
        height = query_img.shape[0]
        train_img_transformed = cv2.warpPerspective(train_img, H, (width, height))
        return train_img_transformed
    
    def imgDiff(self, imgA, imgB, kernel=(5, 5)):
        """Returns a image from the difference of imgA and imgB after a gaussian
        smoothing.
        
        Parameters
        ----------
        imgA : ndarray
            Image.

        imgB : ndarray
            Image.

        kernel : tuple, optional
            Kernel size for Gaussian Smoothing.

        Returns
        -------
        img_diff : ndarray
            imgA - imgB

        """
        imgA_smooth = cv2.GaussianBlur(imgA, kernel, cv2.BORDER_DEFAULT)
        imgB_smooth = cv2.GaussianBlur(imgB, kernel, cv2.BORDER_DEFAULT)
        img_diff = cv2.subtract(imgA, imgB)
        return img_diff
    
    def getImage(self, img):
        img = plt.imread(img)
        return img

    def trim(self, img):
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
        fig, axs = plt.subplots(5, 2, figsize=(20, 40), constrained_layout=True)
        # draw original images
        axs[0, 0].imshow(query_img)
        axs[0, 0].set_title('Query img')
        axs[0, 1].imshow(train_img)
        axs[0, 1].set_title('Train img')
        # draw keypoints
        axs[1, 0].imshow(cv2.drawKeypoints(query_img, query_kpts,
                                           None, color=(0, 255, 0)))
        axs[1, 0].set_title('Query img Keypoints')
        axs[1, 1].imshow(cv2.drawKeypoints(train_img, train_kpts, 
                                           None, color=(0, 255, 0)))
        axs[1, 1].set_title('Train img Keypoints')
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
        img = Image.fromarray(img)
        img.save(path)