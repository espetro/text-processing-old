from typing import Tuple

import numpy as np
import cv2

class Page:
    PAGE_SIZE = (0,0)
    PARAGRAPH_SIZE = (0,0)
    LINE_SIZE = (0,0)
    WORD_SIZE = (0,0)

    @staticmethod
    def resize(image: np.ndarray, sz: Tuple[int]):
        """Resizes the image to a new size. It uses different methods for zooming and shreding."""
        orig_sz = np.product(image.shape)
        new_sz = np.product(sz)
        if new_sz > orig_sz:
            method = cv2.INTER_LANCZOS4
        else:
            method = cv2.INTER_AREA

        return cv2.resize(image, sz, method)

    @staticmethod
    def crop(image: np.ndarray):
        """Given a paper image, it crops the background."""
        pass