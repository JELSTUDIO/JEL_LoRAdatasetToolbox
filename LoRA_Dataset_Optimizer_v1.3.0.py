SCRIPT_VERSION = "1.3.0"
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

        # Initialize MediaPipe modules
        self.mp_face_detection = mp.solutions.face_detection
        
        # --- DUAL DETECTOR SETUP ---
        # Detector 0: Short-range (Large/Close faces) - Fast & Precise
        try:
            self.face_detector_0 = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.3,
                static_image_mode=True
            )
        except TypeError:
            self.face_detector_0 = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.3
            )

        # Detector 1: Long-range (Small/Distant faces) - Higher Recall for small targets
        try:
            self.face_detector_1 = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.3,
                static_image_mode=True
            )
        except TypeError:
            self.face_detector_1 = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.3
            )

        # Variables
        self.source_dir = tk.StringVar()
        self.dest_dir = tk.StringVar()
        self.min_face_scale = tk.DoubleVar(value=0.03)
        self.min_face_dim = tk.IntVar(value=256) 
        self.target_count = tk.IntVar(value=60)
        
        # Track incrementing counters per failure type
        self.dnp_counters = {}
        
        # Status tracking for UI feedback
        self.status_var = tk.StringVar(value="Ready")

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
        tk.Button(src_frame, text="Browse", command=self.browse_dest).pack(side="right")

        tk.Label(self.root, text="--- QUALITY GATE SETTINGS (The 'Trash' Filter) ---", font=header_font).pack(anchor="w", **padding)

        tk.Label(self.root, text="Min Face Scale (% of total image area):").pack(anchor="w", **padding)
        self.scale_slider = tk.Scale(self.root, from_=1, to=50, orient="horizontal", command=self.update_scale_label)
        self.scale_slider.set(3)
        self.scale_slider.pack(fill="x", **padding)
        self.scale_label = tk.Label(self.root, text="Current Scale: 3.0%", font=('Arial', 8, 'italic'))
        self.scale_label.pack()

        tk.Label(self.root, text="Min Face Dimension (Absolute Pixels - Width & Height):", font=('Arial', 9, 'bold'), fg="#d32f2f").pack(anchor="w", **padding)
        self.dim_spin = tk.Spinbox(self.root, from_=32, to=2048, textvariable=self.min_face_dim, increment=32)
        self.dim_spin.pack(fill="x", **padding)
        tk.Label(self.root, text="Ensures faces aren't too low-res for LoRA training.", font=('Arial', 8, 'italic')).pack(anchor="w", **padding)
        
        # Status Label
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=('Arial', 8, 'italic'), fg="blue")
        self.status_label.pack(pady=5)

        tk.Label(self.root, text="--- RANKING SETTINGS (Size-Based Selection) ---", font=header_font).pack(anchor="w", **padding)
        tk.Label(self.root, text="Images are prioritized by face size/scale. Higher scores = closer/bigger faces.", font=('Arial', 9)).pack(anchor="w", **padding)

        tk.Label(self.root, text="Target Number of Best Images to Save:").pack(anchor="w", **padding)
        tk.Scale(self.root, from_=1, to=200, orient="horizontal", variable=self.target_count).pack(fill="x", **padding)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=20)

        self.run_btn = tk.Button(self.root, text="RANK BY SIZE & SELECT", bg="#1565c0", fg="white", 
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

    def get_face_metrics(self, image, detector):
        """
        Performs face detection using the specified detector instance.
        Returns metrics tuple: (face_found, face_scale, yaw_cat, face_w, face_h)
        """
        img_h, img_w, channels = image.shape
        
        # Safely handle RGBA/BGR/Grayscale conversion
        if channels == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif channels == 1:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Try full resolution first
        results = detector.process(image_rgb)
        
        # Fallback for ultra-large images: downscale to max 1024px while preserving aspect ratio.
        if not results.detections and max(img_w, img_h) > 1500:
            scale_factor = min(1.0, 1024 / max(img_w, img_h))
            small_img = cv2.resize(image_rgb, None, fx=scale_factor, fy=scale_factor)
            results = detector.process(small_img)
            
        face_found, face_scale, yaw_cat = False, 0.0, "center"
        face_w, face_h = 0, 0 

        if results.detections:
            face_found = True
            # Sort by area to pick the most prominent face
            sorted_dets = sorted(results.detections, 
                                 key=lambda k: k.location_data.relative_bounding_box.width * k.location_data.relative_bounding_box.height, 
                                 reverse=True)
            detection = sorted_dets[0]
            
            keypoints = detection.location_data.relative_keypoints
            
            eye_l_x = keypoints[0].x
            eye_r_x = keypoints[1].x
            nose_x = keypoints[2].x
            
            eye_dist = eye_r_x - eye_l_x
            if eye_dist != 0:
                ratio = (nose_x - eye_l_x) / eye_dist
                if ratio < 0.4: yaw_cat = "right"
                elif ratio > 0.6: yaw_cat = "left"
                else: yaw_cat = "center"

            bbox = detection.location_data.relative_bounding_box
            x, y = int(bbox.xmin * img_w), int(bbox.ymin * img_h)
            w, h = int(bbox.width * img_w), int(bbox.height * img_h)
            x, y = max(0, x), max(0, y)
            w, h = min(w, img_w - x), min(h, img_h - y)
            face_w, face_h = w, h

            if w > 0 and h > 0:
                face_scale = (w * h) / (img_w * img_h)
            
        return face_found, face_scale, yaw_cat, face_w, face_h

    def get_histogram(self, image):
        # Ensure image is BGR for consistent histogram calculation if needed, 
        # though calcHist handles arrays generally.
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def run_process(self):
        src, dst = self.source_dir.get(), self.dest_dir.get()
        target_n = self.target_count.get()
        min_dim_req = self.min_face_dim.get()

        if not src or not dst:
            messagebox.showerror("Error", "Please select both folders.")
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        image_files = [f for f in os.listdir(src) if f.lower().endswith(valid_extensions)]
        if not image_files:
            messagebox.showerror("Error", "No images found.")
            return

        if not os.path.exists(dst): os.makedirs(dst)

        # Initialize counters and lists
        self.progress["maximum"] = len(image_files)
        hitlist = []
        retry_queue = [] # Images to retry with Model 1
        
        total_failed_dnp = 0
        passed_quality_gate_count = 0
        
        # Counts for reporting
        count_corrupt = 0
        count_pass_unselected = 0

        # --- PASS 1: MODEL 0 (Short-Range) ---
        self.status_var.set("Pass 1: Detecting large/close faces...")
        self.root.update_idletasks()
        
        for i, filename in enumerate(image_files):
            file_path = os.path.join(src, filename)
            img = cv2.imread(file_path)
            
            # FIX 1: Handle Unreadable/Corrupt Files -> Save as DNP "badfile"
            if img is None:
                self.save_dnp(filename, "badfile")
                count_corrupt += 1
                continue

            # Use Model 0
            face_found, face_scale, yaw_cat, f_w, f_h = self.get_face_metrics(img, self.face_detector_0)
            
            passed_gate = True
            failure_reason = None

            # Simplified: Faces are always required. No face detected = automatic gate fail.
            if not face_found:
                passed_gate = False
                failure_reason = "noface"
            else:
                if f_w < min_dim_req or f_h < min_dim_req:
                    passed_gate = False
                    failure_reason = "size"
                elif face_scale < self.min_face_scale.get(): 
                    passed_gate = False
                    failure_reason = "scale"
            
            if passed_gate:
                hist = self.get_histogram(img)
                hitlist.append({'path': file_path, 'scale': face_scale, 'yaw': yaw_cat, 'hist': hist})
                passed_quality_gate_count += 1
            else:
                # FIX 2: Separate retry logic from immediate failure counting
                if not face_found:
                    # No face detected in Pass 1 -> Queue for Retry
                    retry_queue.append({
                        'path': file_path,
                        'img': img,
                        'filename': filename
                    })
                else:
                    # Face found but failed gate (e.g. too small) -> Discard immediately & count
                    total_failed_dnp += 1
                    self.save_dnp(filename, failure_reason or "unknown")

            self.progress["value"] = i + 1
            self.root.update_idletasks()

        # --- PASS 2: MODEL 1 (Long-Range) Retry ---
        if retry_queue:
            self.status_var.set(f"Pass 2: Retrying {len(retry_queue)} images with long-range model...")
            self.root.update_idletasks()
            
            for item in retry_queue:
                img = item['img']
                filename = item['filename']
                
                # Use Model 1
                face_found, face_scale, yaw_cat, f_w, f_h = self.get_face_metrics(img, self.face_detector_1)
                
                passed_gate = True
                failure_reason = None

                # Simplified: Faces are always required. No face detected = automatic gate fail.
                if not face_found:
                    passed_gate = False
                    failure_reason = "noface"
                else:
                    if f_w < min_dim_req or f_h < min_dim_req:
                        passed_gate = False
                        failure_reason = "size"
                    elif face_scale < self.min_face_scale.get(): 
                        passed_gate = False
                        failure_reason = "scale"

                if passed_gate:
                    hist = self.get_histogram(img)
                    hitlist.append({'path': item['path'], 'scale': face_scale, 'yaw': yaw_cat, 'hist': hist})
                    passed_quality_gate_count += 1
                else:
                    total_failed_dnp += 1
                    reason = failure_reason if failure_reason else "noface_retry"
                    self.save_dnp(filename, reason)

        if not hitlist:
            messagebox.showerror("Error", "No images passed the Quality Gate.")
            return

        # STEP 2: Sort by Face Scale (Higher is better)
        hitlist.sort(key=lambda x: x['scale'], reverse=True)

        # STEP 3: Round-Robin Selection for Diversity
        winners = []
        selected_hists = []
        buckets = {'left': [], 'center': [], 'right': [], 'other': []}
        for item in hitlist:
            buckets[item['yaw']].append(item)

        bucket_keys = ['center', 'left', 'right', 'other']
        
        while len(winners) < target_n:
            added_in_this_round = False
            for key in bucket_keys:
                if not buckets[key]: continue
                
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
                        buckets[key].pop(i)
                        added_in_this_round = True
                        break
                    if len(winners) >= target_n: break
                if len(winners) >= target_n: break
            if not added_in_this_round: break

        # STEP 4: Copy Winners
        for item in winners:
            base_name = os.path.basename(item['path'])
            dest_path = os.path.join(dst, base_name)
            counter = 1
            name, ext = os.path.splitext(base_name)
            while os.path.exists(dest_path):
                dest_path = os.path.join(dst, f"{name}_{counter}{ext}")
                counter += 1
            shutil.copy2(item['path'], dest_path)

        # FIX 3: Save "Unselected Winners" (Passed Quality but not chosen as top N diverse)
        self.status_var.set(f"Saving {len(hitlist) - len(winners)} unselected images...")
        self.root.update_idletasks()
        
        for item in hitlist:
            if item not in winners:
                base_name = os.path.basename(item['path'])
                reason = "PASSED_NOT_SELECTED"
                current_count = self.dnp_counters.get(reason, 0) + 1
                self.dnp_counters[reason] = current_count
                
                name_only, ext = os.path.splitext(base_name)
                dnp_filename = f"ZZZZZZ-{reason}_{current_count}{ext}"
                dest_path = os.path.join(dst, dnp_filename)
                
                # Handle collisions
                if os.path.exists(dest_path):
                    safe_ctr = 1
                    while os.path.exists(os.path.join(dst, f"{name_only}-{reason}_{safe_ctr}{ext}")):
                        safe_ctr += 1
                    dest_path = os.path.join(dst, f"{name_only}-{reason}_{safe_ctr}{ext}")
                
                try:
                    shutil.copy2(item['path'], dest_path)
                    count_pass_unselected += 1
                except Exception as e:
                    print(f"[WARNING] Failed to save unselected winner {base_name}: {e}")

        self.status_var.set("Process Complete!")
        
        # --- REPORTING ---
        failed_diversity_count = len(hitlist) - len(winners)

        # FIX 4: Corrected final total calculation. 
        # Categories are now mutually exclusive: Winners + Unselected Passed + DNP Failed + Corrupt = Total Scanned
        messagebox.showinfo("Finished", 
            f"Process Complete!\n\n"
            f"1. TOTAL SCANNED: {len(image_files)}\n"
            f"2. PASSED QUALITY GATE (Pool): {passed_quality_gate_count}\n"
            f"   - Saved as Winners: {len(winners)}\n"
            f"   - Passed but Unselected: {count_pass_unselected}\n"
            f"3. FAILED QUALITY GATE (DNP): {total_failed_dnp}\n"
            f"4. CORRUPT/UNREADABLE: {count_corrupt}\n\n"
            f"FINAL TOTAL SAVED IN FOLDER: {len(winners) + count_pass_unselected + total_failed_dnp + count_corrupt}")
        
        self.progress["value"] = 0

    def save_dnp(self, filename, reason):
        """Saves an image to the destination folder with a DNP prefix."""
        try:
            src_path = os.path.join(self.source_dir.get(), filename)
            if not os.path.exists(src_path): return
            
            current_count = self.dnp_counters.get(reason, 0) + 1
            self.dnp_counters[reason] = current_count
            
            base_name, ext = os.path.splitext(filename)
            dnp_filename = f"ZZZZZZ-DNP_{reason}_{current_count}{ext}"
            dest_path = os.path.join(self.dest_dir.get(), dnp_filename)
            
            if os.path.exists(dest_path):
                name_only, _ = os.path.splitext(dnp_filename)
                safe_ctr = 1
                while os.path.exists(os.path.join(self.dest_dir.get(), f"{name_only}_{safe_ctr}{ext}")):
                    safe_ctr += 1
                dest_path = os.path.join(self.dest_dir.get(), f"{name_only}_{safe_ctr}{ext}")
                
            shutil.copy2(src_path, dest_path)
        except Exception as e:
            print(f"[WARNING] Failed to save DNP file {filename}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LoRAOptimizer(root)
    root.mainloop()
