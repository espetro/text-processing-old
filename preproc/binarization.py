
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

        gray = Binarize.to_grayscale(image)
        w, h = gray.shape

        cei = Binarize.contrast_enhancement(gray, w, h)
        edges = Binarize.detect_edges(gray)
        text_eroded = Binarize.locate_text(cei, edges)

        compute_mat = np.asarray(cei)
        mean_filter = 1 / 121 * np.ones((11,11), np.uint8)
        light_distr = Binarize.estimate_light_distribution(w, h, text_eroded, cei, compute_mat)
        light_distr = cv2.filter2D(
            Binarize.minmax_scale(light_distr), 
            -1,
            mean_filter
        )

        return Binarize.balance_light(cei, light_distr, text_eroded)


    @staticmethod
    # @nb.jit(nopython=False)
    def contrast_enhancement(image, w, h, c=0.3):
        """Computes the Contrast-Enhanced Image (CEI)"""
        # Get histogram
        bins = np.arange(0, 300, 10)
        bins[26] = 255
        hist = np.histogram(image, bins)
        
        # Histogram reducing value (hr)
        sqrt_hw = np.sqrt(w * h)
        for i in range(len(hist[0])):
            if hist[0][i] > sqrt_hw:
                hr = i * 10
                break

        # Compute contrast-enhanced image (CEI)
        cei = (image - (hr + 50 * c)) * 2
        cei[cei > 255] = 255
        cei[cei < 0] = 0
        return cei

    @staticmethod
    def detect_edges(image, threshold=30):
        """Performs edge detection on the image."""
        filtr1 = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3))
        filtr2 = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape((3,3))
        filtr3 = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape((3,3))
        filtr4 = np.array([0,1,2,-1,0,1,-2,-1,0]).reshape((3,3))

        eg1 = np.abs(cv2.filter2D(image, -1, filtr1))
        eg2 = np.abs(cv2.filter2D(image, -1, filtr2))
        eg3 = np.abs(cv2.filter2D(image, -1, filtr3))
        eg4 = np.abs(cv2.filter2D(image, -1, filtr4))
        eg_avg = Binarize.minmax_scale((eg1 + eg2 + eg3 + eg4) / 4)

        # apply threshold binarization for eg_avg. It's hardcoded to 30 (based on the paper)
        h, w = eg_avg.shape
        eg_bin = np.zeros((h, w))
        eg_bin[eg_avg >= threshold] = 255
        
        return eg_bin

    @staticmethod
    def locate_text(cei, eg_bin, threshold=60):
        """Combines CEI and Edge Detection images to pop up text."""
        # apply threshold binarization for cei. It's hardcoded to 60 (based on the paper)
        h, w = cei.shape
        cei_bin = np.zeros((h, w))
        cei_bin[cei >= threshold] = 255
        
        # merge cei_bin and eg_bin
        h, w = eg_bin.shape
        tli = 255 * np.ones((h, w))
        tli[eg_bin == 255] = 0
        tli[cei_bin == 255] = 0

        # return the eroded, merged image
        kernel = np.ones((3,3), np.uint8)
        return cv2.erode(tli, kernel, iterations=1)

    @staticmethod
    @nb.jit(nopython=True)
    def estimate_light_distribution(width, height, erosion, cei, mat):
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
        
        return mat

    @staticmethod
    # @nb.jit(nopython=True)
    def balance_light(cei, ldi, erosion, bl=260):
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
        result[np.where(erosion != 0)] *= 1.5
        result[result < 0] = 0
        result[result > 255] = 255

        return np.asarray(result, dtype=np.uint8)

    @staticmethod
    def to_grayscale(image):
        """"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray.astype(np.float32)

    @staticmethod
    @nb.jit(nopython=True)
    def minmax_scale(image):
        """Performs Min-Max scaling to a given image"""
        diff = np.max(image) - np.min(image)
        result = (image / diff)
        return (result - np.min(result)) * 255.

