import cv2
from matplotlib import pyplot as plt




def display_results(detections, ref, query, query_name):
    for x, y, w, h in detections.values():
        cv2.rectangle(query, (x, y), (x + w, y + h), (0, 255, 0), 15)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    ax1.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Image with bounding boxes
    ax2.imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Detected Changes in {query_name}")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

def display_results_single(detections, query, title):
    for x, y, w, h in detections.values():
        cv2.rectangle(query, (x, y), (x + w, y + h), (0, 255, 0), 15)

    # Original image
    plt.imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()