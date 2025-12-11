import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class CVDSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Vision Deficiency Simulator")
        self.root.geometry("1000x700")
        
        # Accurate Machado-based CVD simulation matrices
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
        
        # Create GUI
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load image button
        ttk.Button(control_frame, text="Load Image", 
                  command=self.load_image).pack(side=tk.LEFT, padx=(0, 20))
        
        # CVD type selection
        ttk.Label(control_frame, text="CVD Type:").pack(side=tk.LEFT, padx=(0, 5))
        
        cvd_frame = ttk.Frame(control_frame)
        cvd_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Radiobutton(cvd_frame, text="Protanopia (Red-blind)", 
                       variable=self.cvd_type, value='protanopia',
                       command=self.process_current_image).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(cvd_frame, text="Deuteranopia (Green-blind)", 
                       variable=self.cvd_type, value='deuteranopia',
                       command=self.process_current_image).pack(side=tk.LEFT)
        
        ttk.Radiobutton(cvd_frame, text="Universal Red-Green", 
                        variable=self.cvd_type, value='universal',
                        command=self.process_current_image).pack(side=tk.LEFT, padx=(0, 10))
        
        # Correction strength
        strength_frame = ttk.Frame(control_frame)
        strength_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(strength_frame, text="Correction Strength:").pack(side=tk.LEFT, padx=(0, 5))
        strength_scale = ttk.Scale(strength_frame, from_=0.0, to=1.0, 
                                  variable=self.correction_strength, length=150,
                                  command=self.on_strength_change)
        strength_scale.pack(side=tk.LEFT, padx=(0, 5))
        self.strength_label = ttk.Label(strength_frame, text="0.6")
        self.strength_label.pack(side=tk.LEFT)
        
        # Process button
        ttk.Button(control_frame, text="Process Image", 
                  command=self.process_current_image).pack(side=tk.LEFT, padx=(20, 0))
        
        # Image display frame
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create image display areas
        self.image_frames = {}
        self.image_labels = {}
        
        titles = ['Original', 'Simulated CVD', 'Daltonized', 'Enhanced & Filtered']
        
        for i, title in enumerate(titles):
            frame = ttk.LabelFrame(image_frame, text=title, padding="5")
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            
            label = ttk.Label(frame, text="No image loaded", anchor="center")
            label.pack(expand=True, fill=tk.BOTH)
            
            self.image_frames[title.lower().replace(' ', '_').replace('&', 'and')] = frame
            self.image_labels[title.lower().replace(' ', '_').replace('&', 'and')] = label
        
        # Configure grid weights for equal distribution
        for i in range(4):
            image_frame.columnconfigure(i, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_text = tk.Text(stats_frame, height=3, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.X)
        
        # Save button frame
        save_frame = ttk.Frame(main_frame)
        save_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(save_frame, text="Save Results", 
                  command=self.save_results).pack(side=tk.RIGHT)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image
                img = cv2.imread(file_path)
                if img is None:
                    messagebox.showerror("Error", "Could not load the image file.")
                    return
                
                # Convert BGR to RGB
                self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Display original image
                self.display_image(self.original_image, 'original')
                
                # Process the image
                self.process_current_image()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def simulate_cvd(self, image, deficiency_type):
        matrix = self.cvd_matrices[deficiency_type]
        img = image.astype(np.float32) / 255.0
        transformed = img @ matrix.T
        transformed = np.clip(transformed, 0, 1)
        return (transformed * 255).astype(np.uint8)
    
    def daltonize(self, image, deficiency_type):
        normal = image.astype(np.float32) / 255.0

        if deficiency_type == 'universal':
            #Simulate Both
            sim_prot = self.simulate_cvd(image, 'protanopia').astype(np.float32) / 255.0
            sim_deut = self.simulate_cvd(image, 'deuteranopia').astype(np.float32) / 255.0
            
            #Calculate error for both
            error_prot = normal - sim_prot
            error_deut = normal - sim_deut

            #Average the error (The Universal "Red-Green" Error)
            error = (error_prot + error_deut) / 2.0
        
        else:
            simulated = self.simulate_cvd(image, deficiency_type).astype(np.float32) / 255.0
            error = normal - simulated
        
        correction = normal + error * self.correction_strength.get()
        correction = np.clip(correction, 0, 1)
        return (correction * 255).astype(np.uint8)
    
    def enhance_contrast(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    def apply_median_filter(self, image):
        return cv2.medianBlur(image, 3)
    
    def compute_statistics(self, image, label):
        mean = np.mean(image)
        std_dev = np.std(image)
        return f"{label} - Mean: {mean:.2f}, Std Dev: {std_dev:.2f}"
    
    def process_current_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        try:
            cvd_type = self.cvd_type.get()
            
            # Processing pipeline
            # Simulated CVD image
            if cvd_type == "universal":
                # For universal, show the average of protanopia + deuteranopia simulation
                sim_prot = self.simulate_cvd(self.original_image, "protanopia")
                sim_deut = self.simulate_cvd(self.original_image, "deuteranopia")
                simulated = ((sim_prot.astype(np.float32) + sim_deut.astype(np.float32)) / 2).astype(np.uint8)
            else:
                simulated = self.simulate_cvd(self.original_image, cvd_type)
            corrected = self.daltonize(self.original_image, cvd_type)
            enhanced = self.enhance_contrast(corrected)
            filtered = self.apply_median_filter(enhanced)
            
            # Display processed images
            self.display_image(simulated, 'simulated_cvd')
            self.display_image(corrected, 'daltonized')
            self.display_image(filtered, 'enhanced_and_filtered')
            
            # Update frame title to show current CVD type
            if cvd_type == "universal":
                self.image_frames['simulated_cvd'].config(text="Simulated (Prot+Deut Mix)")
            else:
                self.image_frames['simulated_cvd'].config(text=f"Simulated {cvd_type.capitalize()}")

            
            # Compute and display statistics
            original_stats = self.compute_statistics(self.original_image, "Original Image")
            filtered_stats = self.compute_statistics(filtered, "Enhanced Image")
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"{original_stats}\n{filtered_stats}\n")
            self.stats_text.insert(tk.END, f"CVD Type: {cvd_type.capitalize()}, Correction Strength: {self.correction_strength.get():.1f}")
            
            # Store processed images for saving
            self.processed_images = {
                'original': self.original_image,
                'simulated': simulated,
                'corrected': corrected,
                'enhanced': enhanced,
                'filtered': filtered
            }
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
    
    def on_strength_change(self, value):
        self.strength_label.config(text=f"{float(value):.1f}")
        if self.original_image is not None:
            self.process_current_image()
    
    def display_image(self, image, image_key):
        # Resize image for display
        height, width = image.shape[:2]
        max_size = 200
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        label = self.image_labels[image_key]
        label.config(image=photo, text="")
        label.image = photo  # Keep a reference to prevent garbage collection
    
    def save_results(self):
        if not hasattr(self, 'processed_images'):
            messagebox.showwarning("Warning", "No processed images to save.")
            return
        
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if save_dir:
            try:
                cvd_type = self.cvd_type.get()
                
                # Save images
                cv2.imwrite(os.path.join(save_dir, f'original.png'), 
                           cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_dir, f'simulated_{cvd_type}.png'), 
                           cv2.cvtColor(self.processed_images['simulated'], cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_dir, f'daltonized_{cvd_type}.png'), 
                           cv2.cvtColor(self.processed_images['corrected'], cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_dir, f'final_{cvd_type}.png'), 
                           cv2.cvtColor(self.processed_images['filtered'], cv2.COLOR_RGB2BGR))
                
                messagebox.showinfo("Success", f"Images saved to {save_dir}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error saving images: {str(e)}")

def main():
    root = tk.Tk()
    
    app = CVDSimulatorGUI(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()