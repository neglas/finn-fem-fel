import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from logger import log_error



class ChangeDetector:
    """
    Detects differences between two images by comparing edges, connected components, 
    and patch-level features.

    Attributes:
        ref (np.ndarray): Reference image in grayscale.
        query (np.ndarray): Query image in grayscale.
        rois (dict): Regions of interest (bounding boxes) detected as differences.
        roi_sources (dict): Mapping of ROI IDs to their source components.
    """

    
    def __init__(self, ref):
        """
        Initialize ChangeDetector with a reference image.

        Args:
            ref (np.ndarray): Reference image (BGR or grayscale)
        """
        self.ref = self.img_to_gray(ref)
        self.query = None
        self.rois = None
        self.roi_sources = None

    #-----SETTERS/GETTERS-------------
    def set_query(self, query):
        self.query = self.img_to_gray(query)
        if self.query.shape != self.ref.shape: 
            message = (f"Shape mismatch when setting query image:"
                       f"query shape={self.query.shape}, ref shape={self.ref.shape}")
            log_error(message)
            raise ValueError(message)
        
    def set_ref(self, ref):
        self.ref = self.img_to_gray(ref)
                
    def update_rois(self, rois):
        self.rois = rois

    def update_roi_sources(self, rois_sources):
        self.roi_sources = rois_sources

    def get_rois(self):
        return self.rois
    
    def get_roi_sources(self):
        return self.roi_sources
    
    def img_to_gray(self, img):
        """
        Convert image to grayscale if it has 3 channels.
        
        Parameters:
            img: np.ndarray, input image (grayscale or BGR)
        
        Returns:
            Grayscale image (2D array)
        """
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            # Already grayscale
            return img
        else:
            message = (f"Unsupported image shape: {img.shape}")
            log_error(message)
            raise ValueError(message)
    
    #----IMAGE PROCESSING---
    def preprocess(self, img):
        """
        Preprocessing images, i.e. noise filtering
        
        :param img: Description
        """
        return img
    
    def preprocess_query(self, blur_kernel, blur_iterations):
        """
        Preprocessing step for query image
        
        """
        self.query = self.gaussian_blur(self.query, blur_kernel,blur_iterations)

    def preprocess_ref(self, blur_kernel, blur_iterations):
        """
        Preprocessing step for ref image
        
        """
        self.ref = self.gaussian_blur(self.ref, blur_kernel,blur_iterations)

    def gaussian_blur(self, img, kernel_size=5, sigma = 0, iterations=1):
        """
        Apply gaussian blur to image
        """

        for _ in range(iterations): 
            img = cv2.GaussianBlur(img, (kernel_size,kernel_size),sigma)

        return img

    def align(self, ref, query):
        """
        Align query image to ref img
        
        :param img: Description
        """
        raise NotImplementedError()

    #---EDGE DETECTION-----
    def find_edges(self, img, method="Canny", **kwargs):
        """
        Apply edge detection to a grayscale image.

        Args:
            img (np.ndarray): Grayscale image (2D)
            method (str): Edge detection method ("Canny", "Sobel", "Laplacian", "Scharr")
            **kwargs: Method-specific parameters

        Returns:
            np.ndarray: Binary edge map (0 or 255)
        """
        if img.ndim != 2:
            message = (f"Input image is not grayscale (2D)")
            log_error(message)
            raise ValueError(message)

        edges = None

        if method == "Canny":
            # Automatic thresholds if not provided
            v1 = kwargs.get("threshold1", int(0.66 * np.median(img)))
            v2 = kwargs.get("threshold2", int(1.33 * np.median(img)))
            edges = cv2.Canny(img, v1, v2)

        elif method == "Sobel":
            # Sobel derivative order
            dx = kwargs.get("dx", 1)
            dy = kwargs.get("dy", 0)
            ksize = kwargs.get("ksize", 3)
            sobel = cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize)
            edges = np.uint8(np.absolute(sobel))
            # Convert to binary
            _, edges = cv2.threshold(edges, kwargs.get("thresh", 50), 255, cv2.THRESH_BINARY)

        elif method == "Laplacian":
            ksize = kwargs.get("ksize", 3)
            lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
            edges = np.uint8(np.absolute(lap))
            _, edges = cv2.threshold(edges, kwargs.get("thresh", 50), 255, cv2.THRESH_BINARY)

        elif method == "Scharr":
            dx = kwargs.get("dx", 1)
            dy = kwargs.get("dy", 0)
            sch = cv2.Scharr(img, cv2.CV_64F, dx, dy)
            edges = np.uint8(np.absolute(sch))
            _, edges = cv2.threshold(edges, kwargs.get("thresh", 50), 255, cv2.THRESH_BINARY)

        else:
            message = (f"Unknown edge detection method: {method}")
            log_error(message)
            raise ValueError(message)

        # Ensure binary 0 or 255
        edges = (edges > 0).astype(np.uint8) * 255
        return edges
    
    #---FILTERS/MATCHERS----

    def filter_cc_by_area(self, cc, min_area = 0, max_area = 999999):
        """
        Filters connected components based on area
        """
        (total_labels, label_ids, values, centroid_ref) = cc
        rois = {}
        for i in range(1, total_labels):
            area = values[i, cv2.CC_STAT_AREA]

            if min_area <= area <= max_area:  
               
                x, y, w, h, area = values[i]

                rois[i] = [x, y, w, h]

        return rois
    
    def get_gradient_magnitude(self, gray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)
    
    def match_patches(self, ref_patch, query_patch, ssim_thresh=0.75):
        """
        Compare two image patches using SSIM on gradient magnitudes.

        Args:
            ref_patch (np.ndarray): Reference image patch (grayscale)
            query_patch (np.ndarray): Query image patch (grayscale)
            ssim_thresh (float): Threshold above which patches are considered matching

        Returns:
            is_match (bool), score (float): Whether patches match, and SSIM score
        """
        if ref_patch.shape != query_patch.shape: 
            message = (f"Query and reference patches must have the same shape, "
                f"got query={query_patch.shape}, ref={ref_patch.shape}")
            log_error(message)
            raise ValueError(message)
        if not(ref_patch.ndim == 2 and  query_patch.ndim == 2): 
            message = (f"Query and reference patches must be grayscale, "
                f"got query={query_patch.ndim}, ref={ref_patch.ndim }")
            log_error(message)
            raise ValueError(message)
    
        # Gradient magnitude
        ref_grad = self.get_gradient_magnitude(ref_patch)
        query_grad = self.get_gradient_magnitude(query_patch)

        # Normalize
        ref_grad = cv2.normalize(ref_grad, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        query_grad = cv2.normalize(query_grad, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        
        # Determine proper win_size
        min_side = min(ref_grad.shape[0], ref_grad.shape[1])
        # Choose largest valid odd win_size <= min_side
        win_size = min(7, min_side)
        if win_size % 2 == 0:
            win_size -= 1
            
        if win_size < 3:
            return False, 0.0
        # Compute SSIM
        score = ssim(ref_grad, query_grad, data_range=255, win_size=win_size)
        return score >= ssim_thresh, score
    
    def orb_match(self, ref, query, nFeatures=500,
              min_matches=5, max_distance=40, min_ratio=0.1):

        """
        Patch matching using ORB features
        """
        
        orb = cv2.ORB_create(nFeatures)

        kp1, des1 = orb.detectAndCompute(ref, None)
        kp2, des2 = orb.detectAndCompute(query, None)

        if des1 is None or des2 is None:
            return False, 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Distance-based filtering (quality control)
        good_matches = [m for m in matches if m.distance <= max_distance]

        # Relative match ratio (robust to patch size / texture)
        min_kp = min(len(kp1), len(kp2))
        match_ratio = len(good_matches) / max(min_kp, 1)

        is_match = (
            len(good_matches) >= min_matches and
            match_ratio >= min_ratio
        )

        return is_match, len(good_matches)
    
    def filter_rois_by_patch_match(self, rois, roi_sources, score_min=0.4, score_max=0.8):
        """
        Filter ROIs by matching patches between ref and query images.
        Keeps original ROI keys to maintain compatibility with downstream functions.

        Parameters:
            rois: dict[str, list[int]] - {roi_id: [x, y, w, h]} with keys like 'ref_1', 'qry_2'
            roi_sources: dict[str, list[str]] - original sources for each ROI
            score_min: float - lower threshold for SSIM ambiguity
            score_max: float - upper threshold for SSIM ambiguity

        Returns:
            filtered_rois: dict[str, list[int]] - only ROIs that differ, keys unchanged
            filtered_roi_sources: dict[str, list[str]] - sources corresponding to filtered ROIs
        """
        filtered_rois = {}
        filtered_roi_sources = {}

        for k, (x, y, w, h) in rois.items():
            patch_ref = self.ref[y:y+h, x:x+w]
            patch_query = self.query[y:y+h, x:x+w]

            is_match, score = self.match_patches(patch_ref, patch_query)

            # If SSIM is ambiguous and patch is small, use ORB
            if score_min < score < score_max and (w < 32 or h < 32):
                result, _ = self.orb_match(patch_ref, patch_query)
                is_match = result

            if not is_match:
                # Keep original keys!
                filtered_rois[k] = [x, y, w, h]
                filtered_roi_sources[k] = roi_sources.get(k, [k])

        return filtered_rois, filtered_roi_sources
    
    #---MERGERS/COMBINERS---
    def merge_dicts_unique_keys(self, dict1, dict2):
        """
        Merge two dicts and make sure keys are unique.
        """
        result = dict(dict1)  # start with dict1
        for key, value in dict2.items():
            new_key = key
            counter = 1
            # keep modifying key until it's unique
            while new_key in result:
                new_key = f"{key}_{counter}"
                counter += 1
            result[new_key] = value
        return result
    
    
        
    def merge_overlapping_rois(self, rois, overlap_mode="iou", overlap_thresh=0.7):
        """
        Merge overlapping ROIs.

        Parameters:
            rois: dict[int, list[int]] - ROI bounding boxes
            overlap_mode: str ("iou" or "fraction") - type of overlap metric
            overlap_thresh: float - minimum fraction/IoU to merge

        Returns:
            merged_boxes: dict[int, list[int]] - merged ROI bounding boxes
            merged_sources: dict[int, list[str]] - combined sources for each merged box
        """
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
            yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = boxA[2]*boxA[3]
            boxBArea = boxB[2]*boxB[3]
            return interArea / float(boxAArea + boxBArea - interArea)

        def overlap_fraction(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
            yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
            inter_area = max(0, xB - xA) * max(0, yB - yA)
            if inter_area == 0: return 0.0
            areaA = boxA[2]*boxA[3]
            areaB = boxB[2]*boxB[3]
            return inter_area / min(areaA, areaB)

        def get_overlap(boxA, boxB, mode="iou"):
            if mode == "iou": return iou(boxA, boxB)
            elif mode == "fraction": return overlap_fraction(boxA, boxB)
            else:
                message = f"Unknown overlap mode: {mode}"
                log_error(message)
                raise ValueError(message)

        boxes_keys = list(rois.keys())
        used = set()
        merged_boxes = {}
        merged_sources = {}
        new_id = 0

        for i, ki in enumerate(boxes_keys):
            if ki in used: continue
            box_i = [int(v) for v in rois[ki]]
            labels_i = [ki]
            sources_i = []  # collect sources

            # Add sources if they exist
            if hasattr(self, "roi_sources") and ki in self.roi_sources:
                sources_i.extend(self.roi_sources[ki])

            for j, kj in enumerate(boxes_keys[i+1:], start=i+1):
                if kj in used: continue
                box_j = [int(v) for v in rois[kj]]
                if get_overlap(box_i, box_j, mode=overlap_mode) >= overlap_thresh:
                    # Merge geometrically
                    x_min = min(box_i[0], box_j[0])
                    y_min = min(box_i[1], box_j[1])
                    x_max = max(box_i[0] + box_i[2], box_j[0] + box_j[2])
                    y_max = max(box_i[1] + box_i[3], box_j[1] + box_j[3])
                    box_i = [x_min, y_min, x_max - x_min, y_max - y_min]
                    labels_i.append(kj)
                    # Merge sources
                    if hasattr(self, "roi_sources") and kj in self.roi_sources:
                        sources_i.extend(self.roi_sources[kj])
                    used.add(kj)

            merged_boxes[new_id] = [int(v) for v in box_i]
            merged_sources[new_id] = list(set(sources_i))  # unique sources
            used.add(ki)
            new_id += 1

        return merged_boxes, merged_sources
    
   
    def merge_moved_objects(self, rois, roi_sources, max_dist=80, area_ratio_thresh=0.5):
        """
        Merge ROIs likely representing the same object moved between reference and query images.

        Args:
            rois (dict[int, list[int]]): ROI bounding boxes
            roi_sources (dict[int, list[str]]): Sources for each ROI
            max_dist (float): Maximum distance for considering ROIs as the same object
            area_ratio_thresh (float): Minimum area ratio between ROIs

        Returns:
            merged (dict[int, list[int]]): Merged ROI bounding boxes
            merged_sources (dict[int, list[str]]): Original sources for each merged ROI
        """
        ref_rois = {}
        qry_rois = {}

        # Separate ROIs by original source
        for k, box in rois.items():
            sources = roi_sources.get(k, [])
            if any(s.startswith("ref_") for s in sources):
                ref_rois[k] = box
            if any(s.startswith("qry_") for s in sources):
                qry_rois[k] = box

        merged = {}
        merged_sources = {}
        used_qry = set()
        new_id = 0

        for rk, (x1, y1, w1, h1) in ref_rois.items():
            cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
            a1 = w1 * h1

            best_match = None
            best_dist = float("inf")

            for qk, (x2, y2, w2, h2) in qry_rois.items():
                if qk in used_qry:
                    continue

                cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
                a2 = w2 * h2
                area_ratio = min(a1, a2) / max(a1, a2)

                if area_ratio < area_ratio_thresh:
                    continue

                dist = np.hypot(cx1 - cx2, cy1 - cy2)
                if dist < best_dist and dist < max_dist:
                    best_dist = dist
                    best_match = qk

            if best_match:
                x2, y2, w2, h2 = qry_rois[best_match]
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)

                merged[new_id] = [
                    int(x_min), int(y_min),
                    int(x_max - x_min), int(y_max - y_min)
                ]

                # Combine sources from both ROIs
                merged_sources[new_id] = roi_sources.get(rk, []) + roi_sources.get(best_match, [])

                used_qry.add(best_match)
                new_id += 1
            else:
                merged[new_id] = [x1, y1, w1, h1]
                merged_sources[new_id] = roi_sources.get(rk, [])
                new_id += 1

        # Add remaining unmatched query ROIs
        for qk, box in qry_rois.items():
            if qk not in used_qry:
                merged[new_id] = box
                merged_sources[new_id] = roi_sources.get(qk, [])
                new_id += 1

        return merged, merged_sources
    
    #---FIND DIFFERENCES MAIN--------
    def find_differences(self, edge_detector:str = "Canny", edge_dilations= 1, merge_mismatch_kernel = 15,min_area=25):

        """
        Detect differences between reference and query images using edge detection.

        Args:
            edge_detector (str): Edge detection method.
            edge_dilations (int): Number of times to dilate edges.
            merge_mismatch_kernel (int): Kernel size for merging mismatch boxes.
            min_area (int): Minimum area for ROI.
        """
        
        #Find and dilate edges
        ref_edges = self.find_edges(self.ref, edge_detector)
        query_edges = self.find_edges(self.query, edge_detector)
        
        
        kernel = np.ones((3,3), np.uint8)
        for _ in range(edge_dilations):
            ref_edges = cv2.dilate(ref_edges, kernel, iterations=1)
            query_edges = cv2.dilate(query_edges, kernel, iterations=1)
        

        ref_bin = (ref_edges > 0).astype(np.uint8)
        query_bin = (query_edges > 0).astype(np.uint8)

        ref_dist = cv2.distanceTransform(1 - query_bin, cv2.DIST_L2, 5)
        query_dist = cv2.distanceTransform(1 - ref_bin, cv2.DIST_L2, 5)

        ref_mismatch_map = (ref_dist > 0) & (ref_bin == 1)
        query_mismatch_map = (query_dist > 0) & (query_bin == 1)

        ref_mismatch_map = (ref_mismatch_map * 255).astype(np.uint8)
        query_mismatch_map = (query_mismatch_map * 255).astype(np.uint8)

        #Merge missmatch boxes
        if merge_mismatch_kernel:

            if merge_mismatch_kernel % 2 == 0:
                message = (f"merge_mismatch_kernel must be an odd integer")
                log_error(message)
                raise ValueError(message)

            if merge_mismatch_kernel <= 1:
                message = (f"merge_mismatch_kernel must be larger than 1")
                log_error(message)
                raise ValueError(message)

            merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_mismatch_kernel, merge_mismatch_kernel))
            ref_mismatch_map = cv2.morphologyEx(ref_mismatch_map, cv2.MORPH_CLOSE, merge_kernel)
            query_mismatch_map = cv2.morphologyEx(query_mismatch_map, cv2.MORPH_CLOSE, merge_kernel)


        ref_cc = cv2.connectedComponentsWithStats(ref_mismatch_map, 4, cv2.CV_8U)
        query_cc = cv2.connectedComponentsWithStats(query_mismatch_map, 4, cv2.CV_8U)
               
        ref_rois = self.filter_cc_by_area(ref_cc, min_area)
        query_rois = self.filter_cc_by_area(query_cc, min_area)

        #add keys to keep track of which component comes from which image
        ref_checked_rois = {f"ref_{k}": v for k, v in ref_rois.items()}
        query_checked_rois = {f"qry_{k}": v for k, v in query_rois.items()}

        #Merge to a single dict
        self.rois = self.merge_dicts_unique_keys(ref_checked_rois, query_checked_rois)
        self.roi_sources = {k: [k] for k in self.rois.keys()}

        return self.rois, self.roi_sources

    