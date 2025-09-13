import os
import random
import shutil
import pandas as pd
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---

# 1. Path settings
ORIGIN_IMAGE_DIR = "datasets/CocoVal2017/origin" 
TRANSFORMED_IMAGE_DIR = "datasets/CocoVal2017/transformed"
OUTPUT_CSV_DIR = "datasets/CocoVal2017/pairs_csv"

# 2. Similar pair generation
NUM_BASE_IMAGES_FOR_IDENTICAL = 1000 
JPEG_QUALITIES = [90, 75, 50] 
SCALING_FACTORS = [0.7, 1.5]
BLUR_RADII = [1, 2] 
ROTATION_ANGLES = [-5, 5] 
GAUSSIAN_NOISE_VAR = 0.01 

# 3. Dissimilar pair generation
NUM_DISTINCT_PAIRS = 50000

# 4. Other settings
RANDOM_SEED = 42

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions: Image Transformations ---

def apply_jpeg_compression(img, quality, save_path):
    """Applies JPEG compression."""
    img.save(save_path, "JPEG", quality=quality)

def apply_scaling(img, factor, save_path):
    """Applies scaling."""
    original_size = img.size
    new_size = (int(original_size[0] * factor), int(original_size[1] * factor))
    resized_img = img.resize(new_size, Image.Resampling.BILINEAR)
    resized_img.save(save_path)

def apply_gaussian_blur(img, radius, save_path):
    """Applies Gaussian blur."""
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    blurred_img.save(save_path)

def apply_rotation(img, angle, save_path):
    """Applies rotation."""
    rotated_img = img.rotate(angle, expand=True, fillcolor='white')
    rotated_img.save(save_path)

def apply_horizontal_flip(img, save_path):
    """Applies horizontal flip."""
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_img.save(save_path)
    
def apply_gaussian_noise(img, save_path, var=0.01):
    """Applies Gaussian noise."""
    img_array = np.array(img).astype(np.float64)
    mean = 0
    sigma = var ** 0.5
    noise = np.random.normal(mean, sigma * 255, img_array.shape)
    noisy_img_array = np.clip(img_array + noise, 0, 255)
    noisy_img = Image.fromarray(noisy_img_array.astype('uint8'))
    noisy_img.save(save_path)


# --- Main Logic ---

def create_directory_if_not_exists(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Directory created: {path}")

def generate_identical_pairs():
    """Generates perceptually identical pairs by applying transformations."""
    logging.info("--- Starting generation of identical pairs ---")
    
    try:
        all_images = [f for f in os.listdir(ORIGIN_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not all_images:
            logging.error(f"Error: No images found in '{ORIGIN_IMAGE_DIR}'.")
            return []
    except FileNotFoundError:
        logging.error(f"Error: Original image directory not found: '{ORIGIN_IMAGE_DIR}'")
        return []

    random.shuffle(all_images)
    base_images = all_images[:NUM_BASE_IMAGES_FOR_IDENTICAL]
    logging.info(f"Randomly selected {len(base_images)} images as a base.")

    identical_pairs = []
    
    def process_base_image(filename):
        pairs = []
        original_path = os.path.join(ORIGIN_IMAGE_DIR, filename)
        try:
            img = Image.open(original_path).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to open or process image {original_path}: {e}")
            return pairs

        base_name, ext = os.path.splitext(filename)

        transformations = []
        for q in JPEG_QUALITIES:
            transformations.append(('jpeg', q))
        for f in SCALING_FACTORS:
            transformations.append(('scale', f))
        for r in BLUR_RADII:
            transformations.append(('blur', r))
        for a in ROTATION_ANGLES:
            transformations.append(('rotate', a))
        transformations.append(('flip', None))
        transformations.append(('noise', GAUSSIAN_NOISE_VAR))

        for trans_type, param in transformations:
            if param is not None:
                transformed_filename = f"{base_name}_{trans_type}_{str(param).replace('.', '_')}{ext}"
            else:
                transformed_filename = f"{base_name}_{trans_type}{ext}"
            
            transformed_path = os.path.join(TRANSFORMED_IMAGE_DIR, transformed_filename)
            
            try:
                if trans_type == 'jpeg':
                    apply_jpeg_compression(img, param, transformed_path)
                elif trans_type == 'scale':
                    apply_scaling(img, param, transformed_path)
                elif trans_type == 'blur':
                    apply_gaussian_blur(img, param, transformed_path)
                elif trans_type == 'rotate':
                    apply_rotation(img, param, transformed_path)
                elif trans_type == 'flip':
                    apply_horizontal_flip(img, transformed_path)
                elif trans_type == 'noise':
                    apply_gaussian_noise(img, transformed_path, var=param)
                
                pairs.append((original_path, transformed_path))
            except Exception as e:
                logging.error(f"Error applying transformation '{trans_type}' (param: {param}) to {filename}: {e}")
        return pairs

    # Use a thread pool to parallelize image processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_base_image, filename) for filename in base_images]
        with tqdm(total=len(futures), desc="Processing base images") as pbar:
            for future in as_completed(futures):
                identical_pairs.extend(future.result())
                pbar.update(1)

    logging.info(f"Successfully generated {len(identical_pairs)} identical pairs.")
    return identical_pairs


def generate_distinct_pairs():
    """Generates perceptually distinct pairs."""
    logging.info("--- Starting generation of distinct pairs ---")
    
    try:
        all_images = [os.path.join(ORIGIN_IMAGE_DIR, f) for f in os.listdir(ORIGIN_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(all_images) < 2:
            logging.error(f"Error: Fewer than 2 images found in '{ORIGIN_IMAGE_DIR}'. Cannot generate distinct pairs.")
            return []
    except FileNotFoundError:
        logging.error(f"Error: Original image directory not found: '{ORIGIN_IMAGE_DIR}'")
        return []

    distinct_pairs = set() 
    
    with tqdm(total=NUM_DISTINCT_PAIRS, desc="Generating distinct pairs") as pbar:
        while len(distinct_pairs) < NUM_DISTINCT_PAIRS:
            pair = random.sample(all_images, 2)
            sorted_pair = tuple(sorted(pair))
            
            if sorted_pair not in distinct_pairs:
                distinct_pairs.add(sorted_pair)
                pbar.update(1)

    logging.info(f"Successfully generated {len(distinct_pairs)} distinct pairs.")
    return list(distinct_pairs)


def save_pairs_to_csv(pairs, filename):
    """Saves a list of image pairs to a CSV file."""
    df = pd.DataFrame(pairs, columns=['image1_path', 'image2_path'])
    output_path = os.path.join(OUTPUT_CSV_DIR, filename)
    df.to_csv(output_path, index=False)
    logging.info(f"Pair information saved to: {output_path}")

def main():
    """Main execution function."""
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Check and create necessary output directories
    create_directory_if_not_exists(TRANSFORMED_IMAGE_DIR)
    create_directory_if_not_exists(OUTPUT_CSV_DIR)

    # Step 1: Generate identical pairs
    identical_pairs = generate_identical_pairs()
    if identical_pairs:
        save_pairs_to_csv(identical_pairs, "identical_pairs.csv")

    # Step 2: Generate distinct pairs
    distinct_pairs = generate_distinct_pairs()
    if distinct_pairs:
        save_pairs_to_csv(distinct_pairs, "distinct_pairs.csv")
    
    logging.info("--- Task completed! ---")

if __name__ == "__main__":
    main()