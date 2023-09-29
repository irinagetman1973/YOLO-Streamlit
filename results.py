from PIL import Image
import numpy as np

class Results:
    
    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None):
        self.orig_img = np.array(orig_img)  # Convert PIL.Image to NumPy array
        self.orig_shape = self.orig_img.shape[:2]
        self.boxes = boxes
        self.masks = masks
        self.probs = probs
        self.keypoints = keypoints
        self.names = names
        self.path = path

    def plot(self):
        # Plotting logic goes here
        pass

    def save_txt(self, txt_file, save_conf=False):
        # Saving logic goes here
        pass

    def save_crop(self, save_dir, file_name):
        # Saving cropped predictions logic goes here
        pass

    def verbose(self):
        # Generating verbose log string logic goes here
        pass

    def tojson(self, normalize=False):
        # Converting the object to JSON format logic goes here
        pass
    
    def assign_probs(self, probs):
        self.probs = probs
    
    pass

from PIL import Image
import numpy as np

class Results:
    
    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None):
        self.orig_img = np.array(orig_img)  # Convert PIL.Image to NumPy array
        self.orig_shape = self.orig_img.shape[:2]
        self.boxes = boxes
        self.masks = masks
        self.probs = probs
        self.keypoints = keypoints
        self.names = names
        self.path = path

    def plot(self):
        # Plotting logic goes here
        pass

    def save_txt(self, txt_file, save_conf=False):
        # Saving logic goes here
        pass

    def save_crop(self, save_dir, file_name):
        # Saving cropped predictions logic goes here
        pass

    def verbose(self):
        # Generating verbose log string logic goes here
        pass

    def tojson(self, normalize=False):
        # Converting the object to JSON format logic goes here
        pass
    
    def assign_probs(self, probs):
        self.probs = probs

