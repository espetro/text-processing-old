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

    @staticmethod
    def illumination_compensation(image, c=0.3, bl=260):
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
        height, width = gray.shape

        # Contrast Enhancement
        hp = Binarize.get_hist(image)
        sqrt_hw = np.sqrt(height * width)
        hr = Binarize.get_hr(hp, sqrt_hw)
        cei = Binarize.get_CEI(gray, hr, c)

        # Edge detection (threshold is hard coded to >= 30, based on the paper)
        eg_bin = Binarize.edge_detection_bin(gray, th=30)

        # Text location (threshold is hard coded to < 60, based on the paper)
        cei_bin = Binarize.img_threshold(cei, th=60, flag=1)
        
        tli = Binarize.merge(eg_bin, cei_bin)
        kernel = np.ones((3,3), np.uint8)
        erosion = cv2.erode(tli, kernel, iterations=1)

        # Light Distribution estimation
        int_img = np.array(cei)
        ratio = int(width / 20)
        for y in range(width):
            if y % ratio == 0 :
                print(int(y / width * 100), "%")
            for x in range(height):
                if erosion[x][y] == 0:
                    x = Binarize.set_intp_img(int_img, x, y, erosion, cei)

        mean_filter = 1 / 121 * np.ones((11,11), np.uint8)
        ldi = cv2.filter2D(Binarize.minmax_scale(int_img), -1, mean_filter)

        # Compute results
        result = np.divide(cei, ldi) * bl
        result[np.where(erosion != 0)] *= 1.5
        result[result < 0] = 0
        result[result > 255] = 255
        
        return (gray, cei, cei_bin, eg_bin, erosion, ldi, result)

    @staticmethod
    def edge_detection_bin(gray: ndarray, th: int=30):
        m1 = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3))
        m2 = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape((3,3))
        m3 = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape((3,3))
        m4 = np.array([0,1,2,-1,0,1,-2,-1,0]).reshape((3,3))

        eg1 = np.abs(cv2.filter2D(gray, -1, m1))
        eg2 = np.abs(cv2.filter2D(gray, -1, m2))
        eg3 = np.abs(cv2.filter2D(gray, -1, m3))
        eg4 = np.abs(cv2.filter2D(gray, -1, m4))
        eg_avg = Binarize.minmax_scale((eg1 + eg2 + eg3 + eg4) / 4)

        eg_bin = Binarize.img_threshold(eg_avg, 30, flag=0)
        return eg_bin

    @staticmethod
    def img_threshold(image: ndarray, th: int, flag: int):
        h, w = image.shape
        image_bin = np.zeros((h,w))
        if flag == 0:
            image_bin[np.where(image >= th)] = 255
        elif flag == 1:
            image_bin[np.where(image < th)] = 255
        return image_bin

    #get histogram
    @staticmethod
    def get_hist(img):
        bins = np.arange(0, 300, 10)
        bins[26] = 255
        hp = np.histogram(img, bins)
        return hp

    #histogram reducing value
    @staticmethod
    def get_hr(hp, sqrt_hw):
        for i in range(len(hp[0])):
            if hp[0][i] > sqrt_hw:
                return i * 10
            
    #get contrast enhenced image
    @staticmethod
    def get_CEI(img, hr, c):
        CEI = (img - (hr + 50 * c)) * 2
        CEI[np.where(CEI > 255)] = 255
        CEI[np.where(CEI < 0)] = 0
        return CEI
                    
    #draw image
    @staticmethod
    def draw(img):
        tmp = img.astype(np.uint8)
        cv2.imshow('image',tmp)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    #get threshold for the avg edge image
    @staticmethod
    def get_th(img, bins):
        hist = np.histogram(img,bins)
        peak_1_index = np.argmax(hist[0])
        peak_2_index = 0
        if peak_1_index == 0:
            peak_2_index += 1
        for i in range(len(hist[0])):
            if hist[0][i] > hist[0][peak_2_index] and i != peak_1_index:
                peak_2_index = i
        peak_1 = hist[1][peak_1_index]
        peak_2 = hist[1][peak_2_index]
        return ((peak_1 + peak_2) / 2), hist

    @staticmethod
    def get_th2(img, bins):
        num = img.shape[0] * img.shape[1]
        hist = np.histogram(img, bins)
        cdf = 0
        for i in range(len(hist[0])):
            cdf += hist[0][i]
            if cdf / num > 0.85:
                return hist[1][i]

    # merge cei and edge map
    @staticmethod
    def merge(edge, cei):
        h = edge.shape[0]
        w = edge.shape[1]
        new_img = 255 * np.ones((h,w))

        new_img[np.where(edge == 255)] = 0
        new_img[np.where(cei == 255)] = 0
        return new_img

    @staticmethod
    def find_end(tli, x, y):
        i = x
        while(i < tli.shape[0] and tli[i][y] == 0):
            i += 1
        return i - 1

    @staticmethod
    def find_mpv(cei, head, end, y):
        h = []
        e = []
        for k in range(5):
            if head - k >= 0:
                h.append(cei[head-k][y])
            if end + k < cei.shape[0]:
                e.append(cei[end + k][y])
        return np.max(h), np.max(e)
        
        
    # set interpolated image
    @staticmethod
    def set_intp_img(img, x, y, tli, cei):
        head = x
        end = Binarize.find_end(tli, x, y)
        n = end - head + 1
        if n > 30:
            return end
        mpv_h, mpv_e = Binarize.find_mpv(cei, head, end, y)
        for m in range(n):
            img[head+m][y] = mpv_h + (m + 1) * ((mpv_e - mpv_h) / n) 
        return end


