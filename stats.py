import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tkinter as tk
from tkinter import filedialog
import os

def select_file(prompt):
    """Opens a file dialog to let the user select an image."""
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(title=prompt, 
                                           filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
    if not file_path:
        print(f"Selection cancelled for: {prompt}")
        return None
    print(f"Selected: {os.path.basename(file_path)}")
    return file_path

def calculate_statistics():
    print("--- Please Select Your Saved Images ---")
    
    # 1. Select the ORIGINAL Image (Input)
    path_orig = select_file("Step 1/4: Select ORIGINAL (original.png)")
    if not path_orig: return

    # 2. Select the FINAL OUTPUT (Daltonized)
    # This is what normal people see (Panel 3)
    path_final = select_file("Step 2/4: Select DALTONIZED (daltonized.png)")
    if not path_final: return

    # 3. Select the SIMULATED BEFORE (Uncorrected)
    # This is the "Problem" view (Panel 2)
    path_sim_before = select_file("Step 3/4: Select SIMULATED UNCORRECTED (simulated.png)")
    if not path_sim_before: return

    # 4. Select the SIMULATED AFTER (Verification)
    # This is the "Solution" view (Panel 4)
    path_sim_after = select_file("Step 4/4: Select SIMULATED CORRECTED (simulated_view.png)")
    if not path_sim_after: return

    print("\nCalculating stats...\n")

    # Load images
    img_orig = cv2.imread(path_orig)
    img_final = cv2.imread(path_final)
    img_sim_before = cv2.imread(path_sim_before)
    img_sim_after = cv2.imread(path_sim_after)

    # Check validity
    if any(img is None for img in [img_orig, img_final, img_sim_before, img_sim_after]):
        print("ERROR: One or more files is not a valid image.")
        return

    # Convert to RGB
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    img_sim_before = cv2.cvtColor(img_sim_before, cv2.COLOR_BGR2RGB)
    img_sim_after = cv2.cvtColor(img_sim_after, cv2.COLOR_BGR2RGB)

    # ===============================================================
    # OBJECTIVE 1: Effectiveness for CVD Users (Contrast Gain)
    # Compare Panel 2 vs Panel 4
    # ===============================================================
    std_before = np.std(img_sim_before)
    std_after = np.std(img_sim_after)
    contrast_gain = ((std_after - std_before) / std_before) * 100

    # ===============================================================
    # OBJECTIVE 2: Visual Quality for Normal Users (SSIM)
    # Compare Panel 1 (Original) vs Panel 3 (Daltonized)
    # ===============================================================
    gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
    gray_final = cv2.cvtColor(img_final, cv2.COLOR_RGB2GRAY)
    
    # Handle slight size mismatches if any
    if gray_orig.shape != gray_final.shape:
        gray_final = cv2.resize(gray_final, (gray_orig.shape[1], gray_orig.shape[0]))

    score_ssim, _ = ssim(gray_orig, gray_final, full=True)

    # ===============================================================
    # PRINT REPORT
    # ===============================================================
    print("="*50)
    print("   RESEARCH DATA REPORT")
    print("="*50)
    
    print(f"OBJECTIVE 1: Effectiveness (CVD Contrast)")
    print(f"  Metric: Standard Deviation Gain")
    print(f"  - Sim. Uncorrected (Panel 2): {std_before:.2f}")
    print(f"  - Sim. Corrected   (Panel 4): {std_after:.2f}")
    print(f"  ------------------------------------------")
    print(f"  RESULT: {contrast_gain:+.2f}% Improvement")
    
    print(f"\nOBJECTIVE 2: Visual Quality (Normal Vision)")
    print(f"  Metric: SSIM (Structural Similarity)")
    print(f"  ------------------------------------------")
    print(f"  RESULT: {score_ssim:.4f}")
    
    if score_ssim >= 0.85:
        print("  STATUS: PASSED (Naturalness Preserved)")
    else:
        print("  STATUS: WARNING (Check Visual Quality)")
    print("="*50)

if __name__ == "__main__":
    calculate_statistics()