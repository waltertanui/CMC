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
if __name__ == "__main__":
    enhance_latent_fingerprint(r"data\example_latent.png")  # Update path as needed
