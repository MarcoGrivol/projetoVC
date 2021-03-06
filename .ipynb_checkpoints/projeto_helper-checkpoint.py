import cv2
import sys
import numpy as np
from os import listdir
from os.path import join, isfile
from sklearn.svm import OneClassSVM
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
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)

    def processGetDiffAllImages(
        self,
        query_imgs,
        query_idxs,
        folder_path='',
        nfeatures=4000, 
        nOctaveLayers=6, 
        edgeThreshold=4, 
        sigma=2.5):

        query_img_0 = query_imgs[0]
        query_img_45L = query_imgs[1]
        query_img_45R = query_imgs[2]

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(query_img_0)
        axs[1].imshow(query_img_45L)
        axs[2].imshow(query_img_45R)
        plt.show()

        labels = np.delete(self.labels, query_idxs)
        color = ('r', 'g', 'b')
        for i, label in enumerate(labels):
            i += 3
            ident, version, rot = label.split('_') # [id, version, rot.jpg]
            rot = rot.split('.')[0] # [rot, jpg]

            train_img = plt.imread(self.imgs[i])
            if rot == '0':
                img_diff = self.processGetDiff(query_img_0, train_img,
                    nfeatures, nOctaveLayers, edgeThreshold, sigma)
            elif rot == '45L':
                img_diff = self.processGetDiff(query_img_45L, train_img,
                    nfeatures, nOctaveLayers, edgeThreshold, sigma)
                pass
            elif rot == '45R':
                img_diff = self.processGetDiff(query_img_45R, train_img,
                    nfeatures, nOctaveLayers, edgeThreshold, sigma)
                pass
            else:
                sys.exit(f'Unknow rotation ({rot}) in idx: {i}. Label: {label}')
            
            img_diff_label = f'{ident}_DIFF{version}_{rot}.jpg'
            self._saveImage(img_diff, join(folder_path, img_diff_label))
            print(img_diff_label, end=' ')
    
    def processGetDiff(
        self, 
        query_img, 
        query_img_mask,
        train_img, 
        plot=False,
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
        query_img_s = cv2.GaussianBlur(query_img, (9, 9), cv2.BORDER_DEFAULT)
        train_img_s = cv2.GaussianBlur(train_img, (9, 9), cv2.BORDER_DEFAULT)
        ret = self.transform(query_img_s, train_img_s, nfeatures, nOctaveLayers, edgeThreshold, sigma)
        train_img_t = ret[0]
        train_img_t = cv2.bitwise_and(train_img_t, query_img_mask)
        
        # query_img_s = cv2.GaussianBlur(query_img, (5, 5), cv2.BORDER_DEFAULT)
        # train_img_s = cv2.GaussianBlur(train_img_t, (5, 5), cv2.BORDER_DEFAULT)
        img_diff = cv2.subtract(train_img_t, query_img_s)
        img_diff_trim = self.trim(img_diff)
        
        if plot:
            query_kpts = ret[1]
            train_kpts = ret[2]
            matches = ret[3]
            self._plot_imgs(query_img, train_img, query_kpts, train_kpts,
                            matches, train_img_t, img_diff_trim)
        return img_diff_trim
    
    def transform(
        self, 
        query_img, 
        train_img,
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
        sift = cv2.SIFT_create(
            nfeatures=nfeatures, 
            nOctaveLayers=nOctaveLayers, 
            edgeThreshold=edgeThreshold, 
            sigma=sigma)
        query_kpts, query_desc = sift.detectAndCompute(query_img, None)
        train_kpts, train_desc = sift.detectAndCompute(train_img, None)
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(train_desc, query_desc)
        matches = sorted(matches, key=lambda x:x.distance)
        
        H = self.getHomography(query_kpts, train_kpts, matches)

        width = query_img.shape[1]
        height = query_img.shape[0]
        train_img_t = cv2.warpPerspective(train_img, H, (width, height))
        return (train_img_t, query_kpts, train_kpts, matches)
    
    def thresholdAllImages(self, imgs_diff_folder, desired_rot=None, threshold_value=50):
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
        imgs, labels = self._getImagesFromFolder(imgs_diff_folder)
        results = []
        for i, img in enumerate(imgs):
            ident, _, rot = labels[i].split('_') # [id, version, rot.jpg]
            rot = rot.split('.')[0] # [rot, jpg]
            if desired_rot == None or rot == desired_rot:
                img_diff = plt.imread(img)
                img_diff_gray = cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY)
                _, img_diff_t = cv2.threshold(img_diff_gray, threshold_value, 1, cv2.THRESH_BINARY)

                t = np.sum(img_diff_t)
                results.append([t, int(ident)])
        return np.array(results)

    def compareHistogramAllImages(
        self,
        query_hist,
        imgs_diff_folder,
        desired_rot=None,
        hist_size=[256],
        ranges=[0, 255]):

        imgs, labels = self._getImagesFromFolder(imgs_diff_folder)
        results = []
        for i, img in enumerate(imgs):
            ident, _, rot = labels[i].split('_') # [id, version, rot.jpg]
            rot = rot.split('.')[0] # [rot, jpg]
            if desired_rot == None or rot == desired_rot:
                img_diff = plt.imread(img)
                
                train_hist = []
                hist_corr = []
                for j in range(3):
                    train_hist.append(cv2.calcHist([img_diff], [j], None, hist_size, ranges))
                    corr = cv2.compareHist(query_hist[j], train_hist[j], cv2.HISTCMP_BHATTACHARYYA)
                    hist_corr.append(corr)
                
                hist_corr.append(int(ident))
                results.append(hist_corr)
        return np.array(results)
    
    def ssimAllImages(self, query_img, query_idx, threshold_value=50):
        query_label = self.labels[query_idx]
        query_rot = query_label.split('_')[2].split('.')[0]
        
        results = []
        for i, label in enumerate(self.labels):
            rot = label.split('_')[2].split('.')[0]
            if rot == query_rot and i != query_idx:
                # rotacao correta e imagem ?? diferente
                img_diff = self.processGetDiff(query_img, plt.imread(self.imgs[i]))
                
                img_diff_gray = cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY)
                query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
                _, img_t = cv2.threshold(img_diff_gray, threshold_value, 1, cv2.THRESH_BINARY)
                
                ssim_value = ssim(query_img_gray, img_t, data_range=img_t.max() - img_t.min())
                results.append([i, ssim_value])
        return np.array(results)
    
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
    
    def predictAndScoreSVM(self, X):
        true_labels = X[:, -1]
        X = X[:, :-1]
        clf = OneClassSVM().fit(X)
        pred = clf.predict(X)
        
        pred_outliers = pred == 1
        true_outliers = true_labels != 0
        pred_x_true = pred_outliers == true_outliers
        
        acc = np.sum(pred_x_true) / len(pred_x_true)
        return acc
            
    def predictAndScoreThreshold(self, X):
        idx = 20
        X_fit = X[:idx, 0]
        X_test = X[idx:, 0]
        true_labels = X[idx:, 1]

        inliers_max_value = np.max(X_fit)
        threshold = inliers_max_value

        pred_outliers = []
        for value in X_test:
            if value > threshold:
                pred_outliers.append(True)
            else:
                pred_outliers.append(False)

        true_outliers = true_labels != 0
        pred_x_true = pred_outliers == true_outliers

        acc = np.sum(pred_x_true) / len(pred_x_true)
        return (acc, threshold)

    def trim(self, img):
        # row, col, d = img.shape
        # # left
        # for i in range(col):
        #     if np.sum(img[:, i, :]) > 0:
        #         break
        # # right
        # for j in range(col - 1, 0, -1):
        #     if np.sum(img[:, j, :]) > 0:
        #         break
        # # up
        # for m in range(row):
        #     if np.sum(img[m, :, :]) > 0:
        #         break
        # # down
        # for n in range(row - 1, 0, -1):
        #     if np.sum(img[n, :, :]) > 0:
        #         break
        m = 680
        n = 2711
        i = 872
        j = 2095
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
        img = Image.fromarray(img)
        img.save(path)