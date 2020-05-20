from skimage.exposure import adjust_sigmoid
from skimage.filters import threshold_sauvola
from numpy import ndarray

import cv2
import numpy as np
import numba as nb

class Binarize:
    """
    Image binarization functions
    Source:
        https://github.com/fanyirobin/text-image-binarization
        https://github.com/arthurflor23/handwritten-text-recognition

    This class uses numba to accelerate numpy-based functions
    """
    np.seterr(divide='ignore', invalid='ignore')

    @staticmethod
    def threshold_sauvola(image, window_sz=3):
        """Computes the sauvola threshold for the given image. Using Scikit-Image implementation.
        This is a faster alternative for the Illumination Compensation algorithm.
        Based in
            J. Sauvola, T. Seppanen, S. Haapakoski, M. Pietikainen,
            Adaptive Document Binarization, in IEEE Computer Society Washington, 1997.

        Source:
            https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_sauvola

        Parameters
        ----------
            image: 3D numpy array
                A RGB image
            window_sz: int, default 3
                Window size for the Sauvola Threshold (look in source)

        Returns
        -------
            2D numpy array, the binarized result image
        """
        gray = Binarize.to_grayscale(image, as_float=False)
        w, h = gray.shape

        cei = Binarize.contrast_enhancement(gray, h, w)
        cei = cei.astype(np.uint8)
        
        threshold = threshold_sauvola(cei, window_size=window_sz)
        result = cei > threshold
        return result.astype(np.uint8) * 255  # from bool [0,1] to int [0, 255]
        
    @staticmethod
    def illumination_compensation(image, c=0.3):
        """Illumination Compensation based in:
            K.-N. Chen, C.-H. Chen, C.-C. Chang,
            Efficient illumination compensation techniques for text images, in
            Digital Signal Processing, 22(5), pp. 726-733, 2012.

        Steps performed:
            1. RGB to Grayscale
            2. Contrast Enhancement
            3. Edge Detection
            4. Text Localization
            5. Lightness Distribution Estimation
            6. Lightness Balancing

        Parameters
        ----------
            image: 3D numpy array
                A RGB image
            c: int, default 0.3
                A parameter in the interval [0.1..0.4] used to further reduce image brightness.
                                
        Returns
        -------
            2D numpy array, the binarized result image
        """
        if image.shape[-1] != 3:
            raise ValueError("Image is not RGB")

        gray = Binarize.to_grayscale(image, as_float=True)
        h, w = gray.shape

        cei = Binarize.contrast_enhancement(gray, h, w, c)
        edges = Binarize.detect_edges(gray)
        text_eroded = Binarize.locate_text(cei, edges)

        compute_mat = np.asarray(cei)
        Binarize.estimate_light_distribution(w, h, text_eroded, cei, compute_mat)
        compute_mat = Binarize.minmax_scale(compute_mat)

        mean_filter = 1 / 121 * np.ones((11,11), np.uint8)
        light_distr = cv2.filter2D(compute_mat, -1, mean_filter)

        return Binarize.balance_light(cei, light_distr, text_eroded)


    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def contrast_enhancement(image: ndarray, h:int, w:int, c:float=0.3):
        """Computes the Contrast-Enhanced Image (CEI)"""
        # Get histogram
        bins = np.arange(0, 300, 10)
        bins[26] = 255
        hist = np.histogram(image, bins)
        
        # Histogram reducing value (hr)
        sqrt_hw = np.sqrt(h * w)
        hr = 0
        for i in range(len(hist[0])):
            if hist[0][i] > sqrt_hw:
                hr = i * 10
                break

        # Compute contrast-enhanced image (CEI)
        cei = (image - (hr + 50 * c)) * 2
        for x in range(h):
            for y in range(w):
                # keep values between 0..255
                if cei[x, y] > 255:
                    cei[x, y] = 255
                elif cei[x, y] < 0:
                    cei[x, y] = 0

        return cei

    @staticmethod
    @nb.jit(forceobj=True, nogil=True)
    def detect_edges(image: ndarray, threshold:int=30):
        """Performs edge detection on the image."""
        filtr1 = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3))
        filtr2 = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape((3,3))
        filtr3 = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape((3,3))
        filtr4 = np.array([0,1,2,-1,0,1,-2,-1,0]).reshape((3,3))

        eg1 = np.abs(cv2.filter2D(image, -1, filtr1))
        eg2 = np.abs(cv2.filter2D(image, -1, filtr2))
        eg3 = np.abs(cv2.filter2D(image, -1, filtr3))
        eg4 = np.abs(cv2.filter2D(image, -1, filtr4))
        eg_avg = Binarize.minmax_scale((eg1 + eg2 + eg3 + eg4) / 4.0)

        # apply threshold binarization for eg_avg. It's hardcoded to 30 (based on the paper)
        eg_bin = Binarize.bin_threshold(eg_avg, threshold=threshold, flag=0)
        return eg_bin

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def bin_threshold(image: ndarray, threshold:int, flag:int):
        """Performs threshold binarization.

        Parameters
        ----------
            flag: int
                Comparison operator. 0 is ">=" (H2H), 1 is "<" (H2L).

        Returns
        -------
            image_bin: ndarray
        """
        h, w = image.shape
        image_bin = np.zeros((h, w))

        if flag == 0: # >=
            for x in range(h):
                for y in range(w):
                    if image[x, y] >= threshold:
                        image_bin[x, y] = 255
        else: # <
            for x in range(h):
                for y in range(w):
                    if image[x, y] < threshold:
                        image_bin[x, y] = 255

        return image_bin

    @staticmethod
    @nb.jit(forceobj=True, nogil=True)
    def locate_text(cei: ndarray, edges_bin: ndarray, threshold:int=60):
        """Combines CEI and Edge Detection images to pop up text."""
        # apply threshold binarization for cei. It's hardcoded to 60 (based on the paper)
        cei_bin = Binarize.bin_threshold(cei, threshold=threshold, flag=1)

        # merge cei_bin and eg_bin
        h, w = edges_bin.shape
        tli = 255 * np.ones((h, w))

        # invert white pixels
        # tli[eg_bin == 255] = 0
        # tli[cei_bin == 255] = 0
        for x in range(h):
            for y in range(w):
                if (edges_bin[x, y] == 255) or (cei_bin[x, y] == 255):
                    tli[x, y] = 0

        # return the eroded, merged image
        kernel = np.ones((3,3), np.uint8)
        return cv2.erode(tli, kernel, iterations=1)

    @staticmethod
    @nb.jit(nopython=True)
    def estimate_light_distribution(width: int, height: int, erosion: ndarray, cei: ndarray, mat: ndarray):
        """Light distribution performed by numba (thanks @Sundrops)"""
        for y in range(width):
            for x in range(height):
                if erosion[x][y] == 0:
                    # Set interpolated image (in "mat")
                    i = x

                    while i < erosion.shape[0] and erosion[i][y] == 0:
                        i += 1

                    end = i - 1
                    n = end - x + 1

                    if n <= 30:
                        h, e = [], []

                        for k in range(5):
                            if x - k >= 0:
                                h.append(cei[x - k][y])

                            if end + k < cei.shape[0]:
                                e.append(cei[end + k][y])

                        mpv_h, mpv_e = max(h), max(e)

                        for m in range(n):
                            mat[x + m][y] = mpv_h + (m + 1) * ((mpv_e - mpv_h) / n)

                    x = end
                    break

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def balance_light(cei: ndarray, ldi: ndarray, erosion: ndarray, bl=260):
        """Balancing Light and Generating Result
        
        Parameters
        ----------
            cei
            ldi
            erosion
            bl: int, default 260
                bl is a luminance-level adjusting parameter in the range [200..300]
        """
        result = np.divide(cei, ldi) * bl
        h, w = result.shape
        for x in range(h):
            for y in range(w):
                if erosion[x, y] != 0:
                    result[x, y] *= 1.5
                # Scale results between 0..255
                if result[x, y] < 0:
                    result[x, y] = 0
                elif result[x, y] > 255:
                    result[x, y] = 255

        return np.asarray(result, dtype=np.uint8)

    @staticmethod
    def to_grayscale(image, as_float=True):
        """"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if as_float:
            return gray.astype(np.float32)
        else:
            return gray.astype(np.uint8)

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def minmax_scale(image):
        """Performs Min-Max scaling to a given image"""
        diff = np.max(image) - np.min(image)
        result = (image / diff)
        return (result - np.min(result)) * 255.

