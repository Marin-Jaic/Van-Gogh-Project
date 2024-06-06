from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import label

def compute_stddev(image, kernel_size):
    # Padding the image to handle borders
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2),
                                    (kernel_size//2, kernel_size//2), (0, 0)), mode='reflect')
    
    stddev = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the neighborhood
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size, :]
            # Calculate the standard deviation in the neighborhood
            stddev[i, j, :] = np.std(neighborhood, axis=(0, 1))
    
    return stddev

def detect_homogeneous_areas(image_path, threshold=2, kernel_size=4):
    # Step 1: Load the image
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_np = np.array(image)

    # Step 2: Apply Gaussian blur to smooth the image
    blurred_image = image.filter(ImageFilter.GaussianBlur(kernel_size))
    blurred_np = np.array(blurred_image)
    
    stddev_image = compute_stddev(blurred_np, kernel_size)
    stddev_gray = np.mean(stddev_image, axis=2)  # Convert to grayscale by averaging RGB stddev
    
    # Step 4: Thresholding
    homogeneous_mask = (stddev_gray < threshold).astype(np.uint8)

    # Step 5: Find contiguous homogeneous regions
    labeled_array, num_features = label(homogeneous_mask)
    
    # Step 6: Create an output image highlighting homogeneous areas
    result_image = image_np.copy()
    for i in range(1, num_features + 1):
        result_image[labeled_array == i] = [255, 0, 0]# Mark homogeneous regions in red

    result_pil = Image.fromarray(result_image)
    homogeneous_mask_pil = Image.fromarray(homogeneous_mask * 255)

    return result_pil, homogeneous_mask_pil

def homogeneous_mask(image_path, threshold=2, kernel_size=4):
    # Step 1: Load the image
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_np = np.array(image)

    # Step 2: Apply Gaussian blur to smooth the image
    blurred_image = image.filter(ImageFilter.GaussianBlur(kernel_size))
    blurred_np = np.array(blurred_image)
    
    stddev_image = compute_stddev(blurred_np, kernel_size)
    stddev_gray = np.mean(stddev_image, axis=2)  # Convert to grayscale by averaging RGB stddev
    
    # Step 4: Thresholding
    homogeneous_mask = (stddev_gray < threshold).astype(np.uint8)

    return homogeneous_mask

def processImages():
    # Example usage
    image_path = "./img/reference_image_resized.jpg"
    #2, 4 or 2,3 seem to work best
    result_image, homogeneous_mask = detect_homogeneous_areas(image_path, threshold=2, kernel_size=4)

    # Save or display the results
    result_image.save('./vangogh/processed_images/homogeneous_areas.png')
    homogeneous_mask.save('./vangogh/processed_images/homogeneous_mask.png')

def main():
    mask = homogeneous_mask("./img/reference_image_resized.jpg")
    sampling_array = np.where(mask == 1)
    print(mask)
    print(list(zip(sampling_array[0], sampling_array[1])))

processImages()
