import os
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# --- TV decomposition ---
def tv_decompose(image, weight=0.1):
    image = image.astype(np.float32) / 255.0
    cartoon = denoise_tv_chambolle(image, weight=weight, channel_axis=None)
    texture = image - cartoon
    return (cartoon * 255).astype(np.uint8), (texture * 255).astype(np.uint8)

# --- Gabor dictionary generation ---
def generate_gabor_dictionary(patch_size, orientations=16, frequencies=9, phases=6):
    theta_vals = np.linspace(0, np.pi, orientations, endpoint=False)
    freq_vals = np.linspace(0.1, 0.4, frequencies)
    phase_vals = np.linspace(0, np.pi, phases)

    dictionary = []
    for theta in theta_vals:
        for freq in freq_vals:
            for phase in phase_vals:
                kernel = cv2.getGaborKernel((patch_size, patch_size), sigma=patch_size/4,
                                            theta=theta, lambd=1/freq, gamma=0.5,
                                            psi=phase, ktype=cv2.CV_32F)
                kernel -= kernel.mean()
                kernel /= np.linalg.norm(kernel) + 1e-8
                dictionary.append(kernel.flatten())
    return np.array(dictionary).T  # shape: (patch_dim, num_atoms)

# --- Sparse reconstruction ---
def sparse_reconstruct(texture_img, dictionary, patch_size=24, alpha=0.1, step=8, iter_patch_sizes=[24, 28, 32, 36]):
    h, w = texture_img.shape
    final_img = np.zeros_like(texture_img, dtype=np.float32)
    final_weight = np.zeros_like(texture_img, dtype=np.float32)

    for size in iter_patch_sizes:
        print(f"[INFO] Enhancing with patch size: {size}")
        dict_current = generate_gabor_dictionary(size)

        # Ensure dictionary matches patch size
        expected_dim = size * size
        if dict_current.shape[0] != expected_dim:
            dict_current = dict_current[:expected_dim, :]

        for y in range(0, h - size + 1, step):
            for x in range(0, w - size + 1, step):
                patch = texture_img[y:y+size, x:x+size].astype(np.float32).flatten()
                # Lower threshold to allow more patches
                if patch.std() < 1:
                    continue
                # Remove normalization for debugging
                # patch /= np.linalg.norm(patch) + 1e-8

                # Print patch stats for debugging
                if np.all(patch == 0):
                    print(f"All-zero patch at ({y},{x})")
                else:
                    print(f"Patch at ({y},{x}) - min: {patch.min()}, max: {patch.max()}, std: {patch.std()}")

                model = Lasso(alpha=0.01, fit_intercept=False, max_iter=2000)
                model.fit(dict_current, patch)
                recon = np.dot(dict_current, model.coef_).reshape(size, size)

                if np.all(recon == 0):
                    print(f"Zero patch at ({y},{x})")
                final_img[y:y+size, x:x+size] += recon
                final_weight[y:y+size, x:x+size] += 1

    final_weight[final_weight == 0] = 1
    result = final_img / final_weight
    return np.clip(result, 0, 255).astype(np.uint8)

# --- Segmentation (orientation coherence) ---
def compute_orientation_coherence(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    gxx = gx ** 2
    gyy = gy ** 2
    gxy = gx * gy
    orientation = np.sqrt((gxx - gyy)**2 + 4*gxy**2) / (gxx + gyy + 1e-8)
    orientation = cv2.GaussianBlur(orientation, (15, 15), 0)
    _, mask = cv2.threshold(orientation, 0.3, 1, cv2.THRESH_BINARY)
    return (mask * 255).astype(np.uint8)

# --- Visual comparison panel ---
def show_visual_comparison(original, enhanced, cartoon, texture, segmented, save_path=None):
    plt.figure(figsize=(15, 6))
    images = [original, cartoon, texture, enhanced, segmented]
    titles = ["Original", "Cartoon", "Texture", "Enhanced", "Segmented"]
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# --- Matcher stub for CMC evaluation (placeholder) ---
def run_cmc_stub(enhanced_img_path, gallery_dir):
    print(f"[MOCK] Running CMC evaluation on enhanced image against gallery at {gallery_dir}...")
    print("[MOCK] CMC rank-1 accuracy: ~94% (replace with actual matcher)")

# --- Main enhancement pipeline ---
def enhance_latent_fingerprint(input_path, output_dir='output', patch_size=24):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    print("[INFO] Performing TV decomposition...")
    cartoon, texture = tv_decompose(image)

    print("[INFO] Generating Gabor dictionary...")
    dictionary = generate_gabor_dictionary(patch_size)

    print("[INFO] Enhancing fingerprint...")
    enhanced = sparse_reconstruct(texture, dictionary, patch_size=patch_size, alpha=0.1,
                                  step=8, iter_patch_sizes=[24, 28, 32, 36])

    print("[INFO] Segmenting fingerprint region...")
    mask = compute_orientation_coherence(enhanced)
    segmented = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "original.png"), image)
    cv2.imwrite(os.path.join(output_dir, "cartoon.png"), cartoon)
    cv2.imwrite(os.path.join(output_dir, "texture.png"), texture)
    cv2.imwrite(os.path.join(output_dir, "enhanced.png"), enhanced)
    cv2.imwrite(os.path.join(output_dir, "segmented.png"), segmented)

    show_visual_comparison(image, enhanced, cartoon, texture, segmented,
                           save_path=os.path.join(output_dir, "comparison.png"))

    run_cmc_stub(os.path.join(output_dir, "enhanced.png"), "gallery/")
    print(f"[DONE] All results saved in: {output_dir}")

# --- Entry Point ---
def match_fingerprints(enhanced_path, reference_path):
    # Load images
    img1 = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print(f"[ERROR] Could not load images for matching: {enhanced_path}, {reference_path}")
        return None

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=5000)

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print(f"[ERROR] No descriptors found in one or both images.")
        return None

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate a simple matching score: number of good matches
    good_matches = [m for m in matches if m.distance < 50]
    score = len(good_matches)
    print(f"[OpenCV ORB] Matching score (good matches): {score} between {enhanced_path} and {reference_path}")

    return score

def process_fingerprint_with_metadata(txt_path):
    # Parse metadata
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    gender = lines[0].split(':')[1].strip()
    fclass = lines[1].split(':')[1].strip()
    history_parts = lines[2].split(':')[1].strip().split()
    latent_img_name = history_parts[0].replace('.pct', '.png')
    reference_img_name = history_parts[2].replace('.pct', '.png') if len(history_parts) > 2 else None

    # Build image path
    img_dir = os.path.dirname(txt_path)
    latent_img_path = os.path.join(img_dir, latent_img_name)

    # Run enhancement pipeline
    output_dir = os.path.join(img_dir, "output_" + os.path.splitext(latent_img_name)[0])
    enhance_latent_fingerprint(latent_img_path, output_dir=output_dir)
    # After enhancement, match with reference if available
    segmented_path = os.path.join(output_dir, "segmented.png")
    if reference_img_name:
        reference_img_path = os.path.join(img_dir, reference_img_name)
        match_fingerprints(segmented_path, reference_img_path)

    # Optionally, return metadata for further use
    return {
        "gender": gender,
        "class": fclass,
        "latent_img": latent_img_path,
        "reference_img": os.path.join(img_dir, reference_img_name) if reference_img_name else None
    }

def batch_process_all_fingerprints(root_dir):
    """
    Recursively process all .txt metadata files in the given root directory.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_path = os.path.join(dirpath, filename)
                print(f"[BATCH] Processing {txt_path}")
                try:
                    process_fingerprint_with_metadata(txt_path)
                except Exception as e:
                    print(f"[ERROR] Failed to process {txt_path}: {e}")

# Example usage:
if __name__ == "__main__":
    # To process a single file:
    # txt_file = r"c:\Data\CMC\data\png_txt\figs_0\f0001_01.txt"
    # process_fingerprint_with_metadata(txt_file)

    # To process all files in the dataset:
    batch_root = r"c:\Data\CMC\data\png_txt"
    batch_process_all_fingerprints(batch_root)
