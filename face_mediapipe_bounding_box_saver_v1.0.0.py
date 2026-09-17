SCRIPT_VERSION = "1.0.0"
APP_NAME = "Face Crop & Resize Tool for LoRA Datasets"
APP_TITLE = f"{APP_NAME} v{SCRIPT_VERSION}"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/MediaPipe info logs

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import glob
import mediapipe as mp

class FaceCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("520x240")
        self.root.resizable(False, False)

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.progress_val = tk.DoubleVar(value=0.0)
        self._stopping = False

        # Initialize BOTH MediaPipe detectors for 2-pass detection
        self.mp_face_0 = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.001
        )
        self.mp_face_1 = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.001
        )

        self._setup_ui()

    def _get_boxes(self, results, w, h):
        """Helper to extract and scale bounding boxes from MediaPipe results."""
        if not results or not results.detections:
            return []
        boxes = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x_rel, y_rel, bw_rel, bh_rel = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            boxes.append((
                int(x_rel * w), 
                int(y_rel * h), 
                max(1, int(bw_rel * w)), 
                max(1, int(bh_rel * h))
            ))
        return boxes

    def _setup_ui(self):
        frame = ttk.Frame(self.root, padding="20")
        frame.grid(row=0, column=0, sticky="nsew")

        # Input Folder
        ttk.Label(frame, text="Input Folder:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.input_dir, width=45).grid(row=0, column=1, padx=5, sticky="we")
        ttk.Button(frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5)

        # Output Folder
        ttk.Label(frame, text="Output Folder:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.output_dir, width=45).grid(row=1, column=1, padx=5, sticky="we")
        ttk.Button(frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5)

        # Status & Progress
        ttk.Label(frame, textvariable=self.status_var).grid(row=3, columnspan=3, sticky="w", pady=(10, 5))
        
        self.progress = ttk.Progressbar(frame, length=490, maximum=100, variable=self.progress_val)
        self.progress.grid(row=4, column=0, columnspan=3, pady=5)

        # Buttons
        self.process_btn = ttk.Button(frame, text="Start Processing", command=self.start_processing)
        self.process_btn.grid(row=6, column=0, padx=(0, 5), pady=(10, 0), sticky="we")

        self.stop_btn = ttk.Button(frame, text="Stop Processing", command=self._stop_processing, state="disabled")
        self.stop_btn.grid(row=6, column=1, padx=(5, 0), pady=(10, 0), sticky="we")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def browse_input(self):
        path = filedialog.askdirectory(title="Select Input Folder")
        if path: self.input_dir.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path: self.output_dir.set(path)

    def _stop_processing(self):
        """Sets abort flag for the processing loop."""
        self._stopping = True
        self.stop_btn.config(state="disabled", text="Stopping...")
        self.status_var.set("Aborting processing...")

    def start_processing(self):
        src_dir = self.input_dir.get().strip()
        dst_dir = self.output_dir.get().strip()

        if not src_dir or not dst_dir:
            messagebox.showwarning("Missing Folders", "Please select both input and output folders.")
            return
        if src_dir == dst_dir:
            messagebox.showerror("Same Folder", "Input and output folders must be different.")
            return

        # 🛡️ ROBUST FILE COLLECTION + DEDUPLICATION
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tiff")
        raw_files = []
        for ext in extensions:
            raw_files.extend(glob.glob(os.path.join(src_dir, ext)))
            raw_files.extend(glob.glob(os.path.join(src_dir, ext.upper())))
        
        unique_files = sorted(set(os.path.normcase(os.path.abspath(f)) for f in raw_files))

        if not unique_files:
            messagebox.showinfo("No Images", "No supported image files found in the input folder.")
            return

        os.makedirs(dst_dir, exist_ok=True)

        # Reset state
        self._stopping = False
        self.total_images = len(unique_files)
        self.processed_count = 0
        self.crops_data = []
        self.progress_val.set(0.0)

        self.process_btn.config(state="disabled")
        self.stop_btn.config(state="normal", text="Stop Processing")
        self.status_var.set(f"Scanning & Detecting faces in {self.total_images} images...")

        # 🔹 SINGLE-CORE SEQUENTIAL PROCESSING
        for i, img_path in enumerate(unique_files):
            if self._stopping:
                break

            try:
                img = cv2.imread(img_path)
                if img is None:
                    self.status_var.set(f"⚠ Warning: Could not read {os.path.basename(img_path)}")
                    continue

                h, w = img.shape[:2]
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 🔄 2-PASS DETECTION: Try model 0 first, fallback to model 1
                results_0 = self.mp_face_0.process(rgb_img)
                boxes = self._get_boxes(results_0, w, h)

                if not boxes:
                    results_1 = self.mp_face_1.process(rgb_img)
                    boxes = self._get_boxes(results_1, w, h)

                if boxes:
                    # Pick largest face per image
                    best_box = max(boxes, key=lambda b: b[2] * b[3])
                    x, y, bw, bh = best_box

                    # 🛡️ CLAMP COORDINATES TO IMAGE BOUNDARIES
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                    
                    if bw <= 0 or bh <= 0:
                        continue

                    # EXACT CROP
                    crop = img[y:y+bh, x:x+bw]
                    total_pixels = bw * bh
                else:
                    # 🆕 NO FACE DETECTED IN EITHER PASS
                    # Mark as 0x0 bounding box with score 0. Keep full original image.
                    crop = img.copy()
                    x, y, bw, bh = 0, 0, 0, 0
                    total_pixels = 0

                self.crops_data.append({
                    'crop': crop,
                    'w': bw, 'h': bh,
                    't': total_pixels,
                    'src_name': os.path.basename(img_path)
                })

            except Exception as e:
                # Log exceptions instead of silently skipping
                print(f"❌ Error processing {os.path.basename(img_path)}: {e}")

            self.processed_count += 1
            percent = (self.processed_count / self.total_images) * 100
            self.progress_val.set(percent)
            self.status_var.set(f"Processing: {self.processed_count}/{self.total_images}")

        # Reset UI controls after loop finishes or aborts
        self.process_btn.config(state="normal")
        self.stop_btn.config(state="disabled", text="Stop Processing")

        if not self.crops_data:
            messagebox.showinfo("No Images Processed", "No images could be read from the selected folder.")
            return

        # ✅ SORT LARGEST TO SMALLEST by total pixel area (w * h)
        # 0-score (no-face) images will naturally fall to the end
        self.crops_data.sort(key=lambda item: item['t'], reverse=True)

        actual_count = len(self.crops_data)
        padding = max(1, len(str(actual_count)))  # Dynamic padding based on total crops

        saved_count = 0
        for rank, item in enumerate(self.crops_data):
            # ✅ FILE NAME FORMAT: {rank}_w{width}h{height}t{total_pixels}.png
            file_name = f"{rank+1:0{padding}d}_w{item['w']}h{item['h']}t{item['t']}.png"
            out_path = os.path.join(dst_dir, file_name)
            
            # ✅ SAVE AS PNG AT MAX COMPRESSION (9 is lossless for PNG format)
            success = cv2.imwrite(out_path, item['crop'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if success:
                saved_count += 1

        self.status_var.set(f"Finished! Saved {saved_count} images.")
        messagebox.showinfo("Done", f"Successfully processed and saved {saved_count} face crops.\nImages are sorted from largest to smallest by bounding-box area. Images without faces are ranked last with a score of 0.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCropApp(root)
    root.mainloop()
