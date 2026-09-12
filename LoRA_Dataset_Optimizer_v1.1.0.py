SCRIPT_VERSION = "1.1.0" # Updated on 20260912
APP_NAME = "LoRA Dataset Optimizer (Rank & Select)"
APP_TITLE = f"{APP_NAME} v{SCRIPT_VERSION}"

import cv2
import mediapipe as mp
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class LoRAOptimizer:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("550x750")
        self.root.minsize(550, 750)

        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        # Variables
        self.source_dir = tk.StringVar()
        self.dest_dir = tk.StringVar()
        self.face_sharpness_threshold = tk.IntVar(value=100) 
        self.require_face = tk.BooleanVar(value=True)
        self.min_face_scale = tk.DoubleVar(value=0.03)
        self.target_count = tk.IntVar(value=60)
        self.ranking_balance = tk.DoubleVar(value=50.0) 

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

        tk.Label(self.root, text="--- QUALITY GATE SETTINGS (The 'Trash' Filter) ---", font=header_font).pack(anchor="w", **padding)

        tk.Label(self.root, text="Min Face Sharpness (Discard anything below this):").pack(anchor="w", **padding)
        tk.Scale(self.root, from_=5, to=300, orient="horizontal", variable=self.face_sharpness_threshold).pack(fill="x", **padding)

        tk.Label(self.root, text="Min Face Scale (Discard tiny faces):").pack(anchor="w", **padding)
        self.scale_slider = tk.Scale(self.root, from_=1, to=50, orient="horizontal", command=self.update_scale_label)
        self.scale_slider.set(3)
        self.scale_slider.pack(fill="x", **padding)
        self.scale_label = tk.Label(self.root, text="Current Scale: 3.0%", font=('Arial', 8, 'italic'))
        self.scale_label.pack()

        tk.Checkbutton(self.root, text="Require Face Presence", variable=self.require_face).pack(anchor="w", **padding)

        tk.Label(self.root, text="--- RANKING SETTINGS (The 'Diversity' Logic) ---", font=header_font).pack(anchor="w", **padding)

        tk.Label(self.root, text="Ranking Priority (0=Sharpness, 100=Size):").pack(anchor="w", **padding)
        self.balance_slider = tk.Scale(self.root, from_=0, to=100, orient="horizontal", variable=self.ranking_balance, command=self.update_balance_label)
        self.balance_slider.set(50)
        self.balance_slider.pack(fill="x", **padding)
        self.balance_label = tk.Label(self.root, text="Equal Weight (Size vs Sharpness)", font=('Arial', 8, 'italic'))
        self.balance_label.pack()

        tk.Label(self.root, text="Target Number of Best Images to Save:").pack(anchor="w", **padding)
        tk.Scale(self.root, from_=1, to=200, orient="horizontal", variable=self.target_count).pack(fill="x", **padding)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=20)

        self.run_btn = tk.Button(self.root, text="RANK BY DIVERSITY & SELECT", bg="#1565c0", fg="white", 
                                 font=('Arial', 12, 'bold'), command=self.run_process)
        self.run_btn.pack(pady=10)

    def update_scale_label(self, val):
        self.min_face_scale.set(float(val) / 100)
        self.scale_label.config(text=f"Current Scale: {float(val):.1f}% of total image area")

    def update_balance_label(self, val):
        v = float(val) / 100
        if v < 0.4: self.balance_label.config(text="Prioritizing Sharpness")
        elif v > 0.6: self.balance_label.config(text="Prioritizing Face Size")
        else: self.balance_label.config(text="Equal Weight (Size vs Sharpness)")

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
        
        face_found, face_sharpness, face_scale, yaw_cat = False, 0.0, 0.0, "center"

        if results.detections:
            face_found = True
            detection = results.detections[0]
            
            # --- FIXED: Use relative_keypoints for FaceDetection model ---
            keypoints = detection.location_data.relative_keypoints
            
            # Keypoint indices for Face Detection: 0=Left Eye, 1=Right Eye, 2=Nose Tip
            # (Note: Indexing may vary slightly by MP version, but 0,1,2 is the standard for detection)
            eye_l_x = keypoints[0].x
            eye_r_x = keypoints[1].x
            nose_x = keypoints[2].x
            
            # Calculate Yaw (ratio of nose position between eyes)
            eye_dist = eye_r_x - eye_l_x
            if eye_dist != 0:
                ratio = (nose_x - eye_l_x) / eye_dist
                if ratio < 0.4: yaw_cat = "right"
                elif ratio > 0.6: yaw_cat = "left"
                else: yaw_cat = "center"

            # Bounding Box Logic
            bbox = detection.location_data.relative_bounding_box
            x, y = int(bbox.xmin * img_w), int(bbox.ymin * img_h)
            w, h = int(bbox.width * img_w), int(bbox.height * img_h)
            x, y = max(0, x), max(0, y)
            w, h = min(w, img_w - x), min(h, img_h - y)

            if w > 0 and h > 0:
                face_crop = image[y:y+h, x:x+w]
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                face_scale = (w * h) / (img_w * img_h)
            
        return face_found, face_sharpness, face_scale, yaw_cat

    def get_histogram(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def run_process(self):
        src, dst = self.source_dir.get(), self.dest_dir.get()
        target_n = self.target_count.get()
        balance = self.ranking_balance.get() / 100.0

        if not src or not dst:
            messagebox.showerror("Error", "Please select both folders.")
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        image_files = [f for f in os.listdir(src) if f.lower().endswith(valid_extensions)]
        if not image_files:
            messagebox.showerror("Error", "No images found.")
            return

        if not os.path.exists(dst): os.makedirs(dst)

        self.progress["maximum"] = len(image_files)
        hitlist = []
        rejected_count = 0

        for i, filename in enumerate(image_files):
            file_path = os.path.join(src, filename)
            img = cv2.imread(file_path)
            if img is None:
                rejected_count += 1
                continue

            face_found, face_sharpness, face_scale, yaw_cat = self.get_face_metrics(img)
            
            passed_gate = True
            if self.require_face.get() and not face_found: passed_gate = False
            elif face_found and face_sharpness < self.face_sharpness_threshold.get(): passed_gate = False
            elif face_found and face_scale < self.min_face_scale.get(): passed_gate = False
            elif not face_found and not self.require_face.get():
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                face_scale = 0.0 
                yaw_cat = "other"

            if passed_gate:
                hist = self.get_histogram(img)
                hitlist.append({'path': file_path, 'scale': face_scale, 'sharp': face_sharpness, 'yaw': yaw_cat, 'hist': hist})
            else:
                rejected_count += 1

            self.progress["value"] = i + 1
            self.root.update_idletasks()

        if not hitlist:
            messagebox.showerror("Error", "No images passed the Quality Gate.")
            return

        # STEP 2: Normalization
        max_s = max(item['sharp'] for item in hitlist) or 1.0
        max_sc = max(item['scale'] for item in hitlist) or 1.0
        if max_s == 0: max_s = 1
        if max_sc == 0: max_sc = 1

        for item in hitlist:
            norm_scale = item['scale'] / max_sc
            norm_sharp = item['sharp'] / max_s
            item['utility'] = (norm_scale * balance) + (norm_sharp * (1.0 - balance))

        # STEP 3: Sort by Utility
        hitlist.sort(key=lambda x: x['utility'], reverse=True)

        # STEP 4: Round-Robin Selection with Redundancy Check
        winners = []
        selected_hists = []
        buckets = {'left': [], 'center': [], 'right': [], 'other': []}
        for item in hitlist:
            buckets[item['yaw']].append(item)

        bucket_keys = ['center', 'left', 'right', 'other']
        
        while len(winners) < target_n:
            added_in_this_round = False
            for key in bucket_keys:
                if not buckets[key]: 
                    continue
                
                # Iterate through the bucket without popping yet
                for i in range(len(buckets[key])):
                    candidate = buckets[key][i]
                    is_duplicate = False
                    
                    for winner_hist in selected_hists:
                        sim = cv2.compareHist(candidate['hist'], winner_hist, cv2.HISTCMP_CORREL)
                        if sim > 0.95:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        winners.append(candidate)
                        selected_hists.append(candidate['hist'])
                        buckets[key].pop(i) # Remove ONLY when successfully added
                        added_in_this_round = True
                        break # Move to next bucket after finding a winner
                    
                    if len(winners) >= target_n: 
                        break
                
                if len(winners) >= target_n: 
                    break
                    
            if not added_in_this_round: 
                break

        # STEP 5: Copy Winners
        for item in winners:
            base_name = os.path.basename(item['path'])
            dest_path = os.path.join(dst, base_name)
            
            # Prevent overwriting
            counter = 1
            name, ext = os.path.splitext(base_name)
            while os.path.exists(dest_path):
                dest_path = os.path.join(dst, f"{name}_{counter}{ext}")
                counter += 1
                
            shutil.copy2(item['path'], dest_path)

        messagebox.showinfo("Finished", 
            f"Process Complete!\n\n"
            f"Total Scanned: {len(image_files)}\n"
            f"Passed Quality Gate: {len(hitlist)}\n"
            f"Rejected: {rejected_count}\n\n"
            f"Diversity Strategy: Round-Robin Angle Selection\n"
            f"Redundancy Strategy: Histogram Correlation Check\n\n"
            f"Final Saved (Top {target_n} Diverse Winners): {len(winners)}")
        
        self.progress["value"] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = LoRAOptimizer(root)
    root.mainloop()
