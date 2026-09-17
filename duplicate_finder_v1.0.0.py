SCRIPT_VERSION = "1.0.0"
APP_NAME = "Exact Image Duplicate Finder"
APP_TITLE = f"{APP_NAME} v{SCRIPT_VERSION}"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import hashlib
from collections import defaultdict
import threading

class ImageDuplicateFinder:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("700x550")
        self.root.minsize(700, 550)
        self.source_dir = tk.StringVar()

        # Optional padding dict for cleaner layout
        self.pad = {"padx": 10, "pady": (5, 10)}
        
        self.setup_ui()

    def setup_ui(self):
        ttk.Label(self.root, text="Exact Image Duplicate Finder", font=("Segoe UI", 12, "bold")).pack(pady=5)

        # --- Folder Selection (Your requested layout) ---
        ttk.Label(self.root, text="SOURCE FOLDER", font=("Segoe UI", 9)).pack(anchor="w", padx=10)
        src_frame = ttk.Frame(self.root)
        src_frame.pack(fill="x", **self.pad)

        self.source_entry = ttk.Entry(src_frame, textvariable=self.source_dir)
        self.source_entry.pack(side="left", expand=True, fill="x")
        ttk.Button(src_frame, text="Browse", command=self.browse_folder).pack(side="right", padx=(5, 0))

        # Scan Button
        self.scan_btn = ttk.Button(self.root, text="🔍 Scan for Duplicates", command=self.start_scan_thread)
        self.scan_btn.pack(**self.pad)

        # Status & Progress
        self.status_var = tk.StringVar(value="Ready. Paste a folder path or click Browse.")
        ttk.Label(self.root, textvariable=self.status_var, foreground="#555").pack(anchor="w", padx=10)

        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill="x", **self.pad)

        # Results Area
        result_frame = ttk.Frame(self.root, padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.text_area = tk.Text(result_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10))
        v_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=v_scroll.set)

        self.text_area.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Images")
        if folder:
            self.source_dir.set(folder)
            self.scan_btn.config(state=tk.NORMAL)
            self.status_var.set(f"✅ Path selected: {folder}")

    def start_scan_thread(self):
        path = self.source_dir.get().strip()
        
        # Validate path immediately
        if not path or not os.path.isdir(path):
            messagebox.showerror("Invalid Path", "Please enter a valid folder path.")
            return
            
        self.scan_btn.config(state=tk.DISABLED)
        self.progress.config(value=0)
        self.status_var.set("Initializing scan...")
        
        # Run in background thread to keep GUI responsive
        threading.Thread(target=self.run_scan, args=(path,), daemon=True).start()

    def run_scan(self, folder):
        try:
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            total = len(files)
            
            if total == 0:
                self.update_ui(text="⚠️ No files found in the selected folder.", progress=100)
                return
                
            hash_map = defaultdict(list)
            processed = 0
            chunk_size = max(1, total // 20)  # Update UI ~every 5% for smooth UX
            
            for fname in files:
                fpath = os.path.join(folder, fname)
                try:
                    with open(fpath, 'rb') as f:
                        h = hashlib.md5()
                        while chunk := f.read(8192):
                            h.update(chunk)
                    hash_map[h.hexdigest()].append(fname)
                except Exception as e:
                    print(f"⚠️ Skipped {fname}: {e}")
                
                processed += 1
                
                # Thread-safe progress update
                if processed % chunk_size == 0 or processed == total:
                    pct = min(100, int((processed / total) * 100))
                    self.update_progress(pct, f"Scanning... {processed}/{total} files")
            
            # Filter only actual duplicates (groups with >1 file)
            duplicates = {h: flist for h, flist in hash_map.items() if len(flist) > 1}
            
            if not duplicates:
                result_text = "🎉 No identical duplicates found. All images are unique."
            else:
                result_text = f"📊 Found {len(duplicates)} group(s) of exact matches:\n{'─' * 40}\n"
                for i, (h, flist) in enumerate(sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True), 1):
                    count = len(flist)
                    result_text += f"🔹 Group {i} ({count} identical file(s)):\n"
                    for f in sorted(flist):
                        result_text += f"   • {f}\n"
                    result_text += "\n"
                
            self.update_ui(text=result_text, progress=100)

        except Exception as e:
            self.update_ui(text=f"❌ Error during scan:\n{str(e)}", progress=100)
        finally:
            self.root.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))

    def update_ui(self, text=None, progress=None):
        """Thread-safe wrapper for results and final status"""
        if text is not None:
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, text)
            self.text_area.config(state=tk.DISABLED)
            
        if progress is not None:
            self.progress.config(value=progress)
            self.status_var.set("Scan complete." if progress == 100 else "Scanning...")

    def update_progress(self, pct, msg):
        """Thread-safe wrapper for intermediate progress updates"""
        self.root.after(0, lambda: (self.progress.config(value=pct), self.status_var.set(msg)))

def main():
    root = tk.Tk()
    app = ImageDuplicateFinder(root)
    root.mainloop()

if __name__ == "__main__":
    main()
