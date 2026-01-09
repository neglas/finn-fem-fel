#CONFIG
REF_SIZE = (640,480)
SHOW_RESULTS = True
REF_PATH = "data/reference/img1.png"
QUERY_PATH = "data/query"
SAVE_FILE = "results.json"

#AUGMENTATIONS
USE_AUGMENTATION = False
NOISE_MEAN = 0
NOISE_SIGMA = 50
NOISE_ITERATIONS = 1
ILLUMINATION_BRIGHTNESS = 1.2
ILLUMINATION_MEAN = 0
ILLUMINATION_SIGMA = 10

#PREPROCESSING
DETECTOR_PREPROCESS = False 
BLUR_KERNEL = 5
BLUR_ITERATIONS = 20

#GOOD ENOUGH (6H)
MERGE_KERNEL_SIZE = 0 #For better merging of bboxes increase
OVERLAP_MODE = "fraction" # switch to "iou" for noisy images
OVERLAP_THRESH = 0.7 # switch to 0.2 for exactly 6 boxes
COMBINE_REGIONS = False #For handling overlapping boxes set to true