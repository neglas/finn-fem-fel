import cv2
from image_loader import load_image, load_images_from_folder
from change_detection import ChangeDetector
from pathlib import Path
from logger import save_detections_to_json, setup_error_log
from draw_utils import display_results, display_results_single
import config
from augmentations import add_gaussian_noise, simulate_illumination

def scale_boxes(rois, scale_x, scale_y):
    scaled_rois = {}
    for k, (x, y, w, h) in rois.items():
        scaled_rois[k] = [
            int(x * scale_x),
            int(y * scale_y),
            int(w * scale_x),
            int(h * scale_y)
        ]
    return scaled_rois


def main():
    
    """
    Find differences between two similar images.
    """

    #Setup
    show_results = config.SHOW_RESULTS
    ref_new_size = config.REF_SIZE
    ref_path = Path(config.REF_PATH)
    query_path = Path(config.QUERY_PATH)
    results_path = Path(config.SAVE_FILE)
    
    #Load ref image
    ref = load_image(ref_path)
    
    #set ref size
    ref = cv2.resize(ref, dsize=ref_new_size,  interpolation=cv2.INTER_LINEAR)
    
    #Load other images
    queries = load_images_from_folder(query_path)
    
    
    #Create detector instance
    detector = ChangeDetector(ref)
    for file_name, query in queries.items():
        detector.update_rois({})
        detector.update_roi_sources({})
        detector.set_ref(ref)

        
        #Augmentations
        if config.USE_AUGMENTATION:
            query = simulate_illumination(query, config.ILLUMINATION_BRIGHTNESS, 
                                          config.ILLUMINATION_MEAN, config.ILLUMINATION_SIGMA)
            query = add_gaussian_noise(query, config.NOISE_MEAN, config.NOISE_SIGMA, 
                                       config.NOISE_ITERATIONS)
        
        original_size = query.shape[:2] 

        if query.shape != ref.shape:
            h, w = ref.shape[:2]
            query = cv2.resize(query, (w,h),  interpolation=cv2.INTER_LINEAR)
        
        detector.set_query(query)

        if config.DETECTOR_PREPROCESS:
            detector.preprocess_query(config.BLUR_KERNEL, config.BLUR_ITERATIONS)
            detector.preprocess_ref(config.BLUR_KERNEL, config.BLUR_ITERATIONS)
        

        detections, sources = detector.find_differences(merge_mismatch_kernel=config.MERGE_KERNEL_SIZE)
        detections, sources = detector.filter_rois_by_patch_match(detections, sources)

        #Combine bounding boxes
        if config.COMBINE_REGIONS:
            detections, sources = detector.merge_overlapping_rois(detections, overlap_mode=config.OVERLAP_MODE, overlap_thresh= config.OVERLAP_THRESH)
            detections, sources = detector.merge_moved_objects(detections, sources)
       
        
        

        scale_x, scale_y = [original_size[0]/ref_new_size[0], original_size[1]/ref_new_size[1]] 
        detections = scale_boxes(detections, scale_x, scale_y)

        #Serialize for parsing
        detections_serializable = {k: [int(vv) for vv in v] for k, v in detections.items()}
        save_detections_to_json(file_name, detections_serializable, results_path)
        
        #Display results on oringinal images 
        if show_results and not config.USE_AUGMENTATION:
            display_results(detections, load_image(ref_path), load_image(Path(query_path/file_name)), file_name)
            display_results_single(detections, load_image(Path(query_path/file_name)), f"Result of differences on {file_name}")
        else:
            #Display result on original image size after augmentations
            display_results(detections, load_image(ref_path), cv2.resize(query, original_size, interpolation=cv2.INTER_LINEAR),  file_name)

if __name__ == "__main__":
    base_path = Path(__file__).parent
    error_log_path = base_path / "output"/"logs"/"errors.log"
    setup_error_log(error_log_path)
    main()

