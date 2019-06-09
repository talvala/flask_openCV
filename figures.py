import numpy as np
import cv2 as cv
from urllib2 import urlopen, URLError
import json


class DetectionManager(object):

    def run(self, image_url, json_path):
        img = self.download_image(image_url)
        final_mask = self.process_image(image_url, json_path)
        return img * final_mask[:, :, np.newaxis]

    @staticmethod
    def process_figure(img, coordinates):
        """
        get image_url, download it, extract figure and return the mask
        :param img: <ndarray>
        :param coordinates: <tuple> (left, top, width, height)
        :return: <list<list>> openCV mask
        """
        mask = np.zeros(img.shape[:2], np.uint8)
        left, top, width, height = coordinates
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (left, top, width, height)
        cv.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        return mask2

    @staticmethod
    def download_image(url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        try:
            resp = urlopen(url)
        except URLError:
            return None
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        # return the image
        return image

    @staticmethod
    def convert_coordinates(json_filepath, whole_body=False):
        converted_coordinates = list()
        with open(json_filepath, 'r') as f:
            data_as_dict = json.load(f)
        # Todo - should_get_whole_body
        # whole_body = should_get_whole_body(data_as_dict)
        try:
            if whole_body:
                figures = data_as_dict["cropHintsAnnotation"]['cropHints']
            else:
                figures = data_as_dict["faceAnnotations"]
        except KeyError:
            return None

        for figure in figures:
            try:
                rect_coordinates = figure['boundingPoly']['vertices']
            except KeyError:
                continue
            top_left, top_right, bottom_right, bottom_left = rect_coordinates
            left = top_left.get("x") or 0
            width = top_right.get("x") - left
            top = top_left.get("y") or 0
            height = bottom_right.get("y") - top
            converted_coordinates.append((left, top, width, height))
        return converted_coordinates

    def process_image(self, image_url, boxes_url):
        img = self.download_image(image_url)
        if img is None:
            return None
        figure_boxes = self.convert_coordinates(boxes_url)
        if not figure_boxes:
            return None
        final_mask = np.zeros(img.shape[:2], np.uint8)
        for box in figure_boxes:
            figure_mask = self.process_figure(img, box)
            final_mask = np.bitwise_or(final_mask, figure_mask)

        return final_mask


# image_url = "http://scd.en.rfi.fr/sites/english.filesrfi/imagecache/rfi_16x9_1024_578/sites/images.rfi.fr/files/aefimagesnew/aef_image/2019-03-28t100320z_719753758_rc19f0b34600_rtrmadp_3_algeria-protests-bouteflika.jpg"
# img = download_image(image_url)
# json_url = "json.json"
#
# final_mask = process_image(image_url, json_url)
# img = img*final_mask[:, :, np.newaxis]
# plt.imshow(img), plt.colorbar(), plt.show()