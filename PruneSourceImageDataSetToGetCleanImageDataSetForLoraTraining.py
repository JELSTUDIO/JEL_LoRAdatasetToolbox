import cv2
import mediapipe as mp
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class LoRAOptimizer:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRA Dataset Optimizer (Rank & Select)")
        self.root.geometry("550x800") # Increased height for new slider
        self.root.geometry("550x800")

        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        # Variables
        self.source_dir = tk.StringVar()
        self.dest_dir = tk.StringVar()
        self.face_sharpness_threshold = tk.IntVar(value=120) 
        self.require_face = tk.BooleanVar(value=True)
        self.min_face_scale = tk.DoubleVar(value=0.03)
        self.target_count = tk.IntVar(value=60) # NEW: Target number of images

        self.setup_ui()

    def setup_ui(self):
        padding = {'padx': 15, 'pady': 5}
        header_font = ('Arial', 10, 'bold')

        # --- Folder Selection ---
        tk.Label(self.root, text="SOURCE FOLDER", font=header_font).pack(anchor="w", **padding)
        src_frame = tk.Frame(self.root)
        src_frame.pack(fill="x", **padding)
        tk.Entry(src_frame, textvariable=self.source_dir).pack(side="left", expand=True, fill="x")
        tk.Button(src_frame, text="Browse", command=self.browse_source).pack(side="right")

        tk.Label(self.root, text="DESTINATION FOLDER", font=header_font).pack(anchor="w", **padding)
        dst_frame = tk.Frame(self.root)
        dst_frame.pack(fill="x", **padding)
        tk.Entry(dst_frame, textvariable=self.dest_dir).pack(side="left", expand=True, fill="x")
        tk.Button(dst_frame, text="Browse", command=self.browse_dest).pack(side="right")

        tk.Label(self.root, text="--- QUALITY GATE SETTINGS ---", font=header_font).pack(anchor="w", **padding)

        # --- Sharpness ---
        tk.Label(self.root, text="Min Face Sharpness (Discard anything below this):").pack(anchor="w", **padding)
        tk.Scale(self.root, from_=5, to=300, orient="horizontal", variable=self.face_sharpness_threshold).pack(fill="x", **padding)

        # --- Face Scale ---
        tk.Label(self.root, text="Min Face Scale (Discard tiny faces):").pack(anchor="w", **padding)
        self.scale_slider = tk.Scale(self.root, from_=1, to=50, orient="horizontal", command=self.update_scale_label)
        self.scale_slider.set(3)
        self.scale_slider.pack(fill="x", **padding)
        self.scale_label = tk.Label(self.root, text="Current Scale: 3.0%", font=('Arial', 8, 'italic'))
        self.scale_label.pack()

        # --- Face Requirement ---
        tk.Checkbutton(self.root, text="Require Face Presence", variable=self.require_face).pack(anchor="w", **padding)

        tk.Label(self.root, text="--- SELECTION SETTINGS ---", font=header_font).pack(anchor="w", **padding)

        # --- NEW: Target Count Slider ---
        tk.Label(self.root, text="Target Number of Best Images to Save:").pack(anchor="w", **padding)
        tk.Scale(self.root, from_=1, to=200, orient="horizontal", variable=self.target_count).pack(fill="x", **padding)

        # --- Progress ---
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=20)

        # --- Run Button ---
        self.run_btn = tk.Button(self.root, text="RANK AND SELECT BEST", bg="#2e7d32", fg="white", 
                                 font=('Arial', 12, 'bold'), command=self.run_process)
        self.run_btn.pack(pady=10)

    def update_scale_label(self, val):
        self.min_face_scale.set(float(val) / 100)
        self.scale_label.config(text=f"Current Scale: {float(val):.1f}% of total image area")

    def browse_source(self):
        path = filedialog.askdirectory()
        if path: self.source_dir.set(path)

    def browse_dest(self):
        path = filedialog.askdirectory()
        if path: self.dest_dir.set(path)

    def get_face_metrics(self, image):
        img_h, img_w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(image_rgb)
        
        face_found = False
        face_sharpness = 0.0
        face_scale = 0.0

        if results.detections:
            face_found = True
            bbox = results.detections[0].location_data.relative_bounding_box
            x, y = int(bbox.xmin * img_w), int(bbox.ymin * img_h)
            w, h = int(bbox.width * img_w), int(bbox.height * img_h)
            x, y = max(0, x), max(0, y)
            w, h = min(w, img_w - x), min(h, img_h - y)

            if w > 0 and h > 0:
                face_crop = image[y:y+h, x:x+w]
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                face_scale = (w * h) / (img_w * img_h)
            
        return face_found, face_sharpness, face_scale

    def run_process(self):
        src = self.source_dir.get()
        dst = self.dest_dir.get()
        target_n = self.target_count.get()

        if not src or not dst:
            messagebox.showerror("Error", "Please select both folders.")
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        image_files = [f for f in os.listdir(src) if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            messagebox.showerror("Error", "No images found.")
            return

        if not os.path.exists(dst):
            os.makedirs(dst)

        self.progress["maximum"] = len(image_files)
        
        # This is our "Hitlist": stores tuples of (file_path, sharpness_score)
        hitlist = []
        total_scanned = 0
        rejected_count = 0

        for i, filename in enumerate(image_files):
            file_path = os.path.join(src, filename)
            img = cv2.imread(file_path)

            if img is None:
                rejected_count += 1
                continue

            face_found, face_sharpness, face_scale = self.get_face_metrics(img)
            
            # --- THE QUALITY GATE ---
            passed_gate = True
            if self.require_face.get() and not face_found:
                passed_gate = False
            elif face_found and face_sharpness < self.face_sharpness_threshold.get():
                passed_gate = False
            elif face_found and face_scale < self.min_face_scale.get():
                passed_gate = False
            elif not face_found and not self.require_face.get():
                # If face isn't required, we still need some sharpness metric 
                # to rank the image. We'll use global sharpness as fallback.
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            if passed_gate:
                hitlist.append((file_path, face_sharpness))
                total_scanned += 1
            else:
                rejected_count += 1

            self.progress["value"] = i + 1
            self.root.update_idletasks()

        # --- THE RANKING AND SELECTION ---
        # Sort hitlist by the second element of the tuple (sharpness) in descending order
        hitlist.sort(key=lambda x: x[1], reverse=True)

        # Take the top N
        final_selection = hitlist[:target_n]

        # Copy the winners
        for file_path, score in final_selection:
            shutil.copy2(file_path, os.path.join(dst, os.path.basename(file_path)))

        messagebox.showinfo("Finished", 
            f"Process Complete!\n\n"
            f"Total Scanned: {len(image_files)}\n"
            f"Passed Quality Gate: {len(hitlist)}\n"
            f"Rejected (Low Quality): {rejected_count}\n\n"
            f"Final Saved (Top {target_n} selected): {len(final_selection)}")
        
        self.progress["value"] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = LoRAOptimizer(root)
    root.mainloop()
