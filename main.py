import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class CVDSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Vision Deficiency Simulator")
        self.root.geometry("1200x600")

        # Machado-based RGB simulation matrices
        self.cvd_matrices = {
            'protanopia': np.array([
                [0.152286, 1.052583, -0.204868],
                [0.114503, 0.786281, 0.099216],
                [-0.003882, -0.048116, 1.051998]
            ]),
            'deuteranopia': np.array([
                [0.367322, 0.860646, -0.227968],
                [0.280085, 0.672501, 0.047413],
                [-0.011820, 0.042940, 0.968881]
            ])
        }

        # Variables
        self.original_image = None
        self.cvd_type = tk.StringVar(value='protanopia')
        self.correction_strength = tk.DoubleVar(value=0.6)

        self.create_widgets()

    # ------------------------- UI SETUP -------------------------
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X)

        ttk.Button(control_frame, text="Load Image",
                   command=self.load_image).pack(side=tk.LEFT, padx=10)

        ttk.Label(control_frame, text="CVD Type:").pack(side=tk.LEFT)
        cvd_frame = ttk.Frame(control_frame)
        cvd_frame.pack(side=tk.LEFT, padx=10)

        ttk.Radiobutton(cvd_frame, text="Protanopia",
                        variable=self.cvd_type, value='protanopia',
                        command=self.process_current_image).pack(side=tk.LEFT)

        ttk.Radiobutton(cvd_frame, text="Deuteranopia",
                        variable=self.cvd_type, value='deuteranopia',
                        command=self.process_current_image).pack(side=tk.LEFT)

        ttk.Radiobutton(cvd_frame, text="Universal",
                        variable=self.cvd_type, value='universal',
                        command=self.process_current_image).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Strength:").pack(side=tk.LEFT, padx=5)
        strength_scale = ttk.Scale(control_frame, from_=0, to=1,
                                   variable=self.correction_strength,
                                   length=150,
                                   command=self.on_strength_change)
        strength_scale.pack(side=tk.LEFT)
        self.strength_label = ttk.Label(control_frame, text="0.6")
        self.strength_label.pack(side=tk.LEFT)

        ttk.Button(control_frame, text="Process",
                   command=self.process_current_image).pack(side=tk.LEFT, padx=20)

        # ----------------- IMAGE DISPLAY -----------------
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.image_labels = {}
        titles = [
            ("Original", "original"),
            ("Simulated CVD (Uncorrected)", "simulated_cvd"),
            ("Daltonized (Corrected)", "daltonized"),
            ("Simulated CVD (Corrected)", "simulated_view")
        ]

        BOX_W = 250
        BOX_H = 250

        for i, (title, key) in enumerate(titles):
            frame = ttk.LabelFrame(image_frame, text=title)
            frame.grid(row=0, column=i, padx=5, sticky="nsew")
            lbl = tk.Label(frame, text="No Image", bg="#f0f0f0", width=BOX_W, height=BOX_H, anchor="center")
            lbl.pack(expand=True, fill=tk.BOTH)
            self.image_labels[key] = lbl

        for i in range(len(titles)):
            image_frame.columnconfigure(i, weight=1)
        
        image_frame.rowconfigure(0, weight=1)


        # Statistics box
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=10)
        self.stats_text = tk.Text(stats_frame, height=4)
        self.stats_text.pack(fill=tk.X)

        # Save button
        ttk.Button(main_frame, text="Save Results",
                   command=self.save_results).pack(anchor="e")

    # ------------------------- FILE LOADING -------------------------
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.display_image(self.original_image, "original")
        self.process_current_image()

    # ------------------------- SIMULATION -------------------------
    def simulate_cvd(self, image, deficiency):
        matrix = self.cvd_matrices[deficiency]
        img = image.astype(np.float32) / 255
        result = np.dot(img, matrix.T)
        return np.clip(result * 255, 0, 255).astype(np.uint8)

    # LMS system for universal simulation
    def _simulate_cvd_lms(self, image, deficiency):
        rgb2lms = np.array([
            [0.31399022, 0.63951294, 0.04649755],
            [0.15537241, 0.75789446, 0.08670142],
            [0.01775239, 0.10944209, 0.87256922]
        ])
        lms2rgb = np.linalg.inv(rgb2lms)

        LMS = image @ rgb2lms.T

        prot_loss = np.array([[0, 1.05118294, -0.05116099]] * 3)
        deut_loss = np.array([[1, 0, 0],
                              [0.9513092, 0, 0.04866992],
                              [0.9513092, 0, 0.04866992]])

        if deficiency == "protanopia":
            LMS_sim = LMS @ prot_loss.T
        else:
            LMS_sim = LMS @ deut_loss.T

        return np.clip(LMS_sim @ lms2rgb.T, 0, 1)

    # Daltonization
    def daltonize(self, image, cvd_type):
        normal = image.astype(np.float32) / 255

        if cvd_type == "universal":
            sim_p = self._simulate_cvd_lms(normal, "protanopia")
            sim_d = self._simulate_cvd_lms(normal, "deuteranopia")
            sim = (sim_p + sim_d) / 2
        else:
            sim = self.simulate_cvd(image, cvd_type).astype(np.float32) / 255

        error = normal - sim
        corrected = np.clip(normal + error * self.correction_strength.get(), 0, 1)
        return (corrected * 255).astype(np.uint8)

    # ------------------------- MAIN PROCESSING -------------------------
    def process_current_image(self):
        if self.original_image is None:
            return

        try:
            cvd_type = self.cvd_type.get()
            original = self.original_image
            original_float = original.astype(np.float32) / 255

            # Simulated (uncorrected)
            if cvd_type == "universal":
                sim_p = self._simulate_cvd_lms(original_float, "protanopia")
                sim_d = self._simulate_cvd_lms(original_float, "deuteranopia")
                simulated = ((sim_p + sim_d) / 2 * 255).astype(np.uint8)
            else:
                simulated = self.simulate_cvd(original, cvd_type)

            # Daltonized (Correction)
            daltonized = self.daltonize(original, cvd_type)

            # Simulated Corrected View
            if cvd_type == "universal":
                d_float = daltonized.astype(np.float32) / 255
                sim_p2 = self._simulate_cvd_lms(d_float, "protanopia")
                sim_d2 = self._simulate_cvd_lms(d_float, "deuteranopia")
                simulated_view = ((sim_p2 + sim_d2) / 2 * 255).astype(np.uint8)
            else:
                simulated_view = self.simulate_cvd(daltonized, cvd_type)

            # Display
            self.display_image(simulated, "simulated_cvd")
            self.display_image(daltonized, "daltonized")
            self.display_image(simulated_view, "simulated_view")

            # Stats
            self.update_stats(original, daltonized, cvd_type)

            # Internal save
            self.processed_images = {
                "original": original,
                "simulated": simulated,
                "daltonized": daltonized,
                "simulated_view": simulated_view
            }

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------- STATS -------------------------
    def compute_stats(self, img):
        return np.mean(img), np.std(img)

    def update_stats(self, original, daltonized, cvd_type):
        m1, s1 = self.compute_stats(original)
        m2, s2 = self.compute_stats(daltonized)

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END,
                               f"Original - Mean: {m1:.2f}, StdDev: {s1:.2f}\n")
        self.stats_text.insert(tk.END,
                               f"Daltonized - Mean: {m2:.2f}, StdDev: {s2:.2f}\n")
        self.stats_text.insert(tk.END,
                               f"CVD Type: {cvd_type.capitalize()}, Strength: {self.correction_strength.get():.2f}")

    # ------------------------- HELPERS -------------------------
    def on_strength_change(self, value):
        self.strength_label.config(text=f"{float(value):.2f}")
        if self.original_image is not None:
            if hasattr(self, "_debounce"):
                self.root.after_cancel(self._debounce)
            self._debounce = self.root.after(180, self.process_current_image)

    def display_image(self, image, key):
        h, w = image.shape[:2]
        max_size = 220
        scale = max_size / max(h, w)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        photo = ImageTk.PhotoImage(Image.fromarray(resized))

        lbl = self.image_labels[key]
        lbl.config(image=photo, text="", anchor="center")
        lbl.image = photo

    # ------------------------- SAVE -------------------------
    def save_results(self):
        if not hasattr(self, "processed_images"):
            messagebox.showwarning("Warning", "No processed images to save.")
            return

        folder = filedialog.askdirectory()
        if not folder:
            return

        for name, img in self.processed_images.items():
            cv2.imwrite(os.path.join(folder, f"{name}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        messagebox.showinfo("Saved", f"Saved to {folder}")

# ------------------------- MAIN -------------------------
def main():
    root = tk.Tk()
    CVDSimulatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
