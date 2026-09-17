# JEL_LoRAdatasetToolbox

A collection of various tools related to creating or pruning or working with data-sets needed for creating LoRA models.

## Available tools
List of currently available tools (Which may contain bugs or errors I haven't discovered yet. I use AI to assist with the coding and bug-hunting):

- **LoRA_Dataset_Optimizer**:
   - version 1.3.0
   - Code-restructure to make image-selections much less fragile and to fix some errors that resulted in incorrect selections previously.
   - version 1.1.0
   - This update attempts to make a more diversified image-selection where it looks for both face-size and face-sharpness and face-angle. The goal is to have the chosen images be those that will create the most flexible LoRA.
   - version 1.0.0
   - Reads a source-folder of assorted images of a person you want to train a LoRA on, and then looks for suitable faces and ranks all images after sharpness, and then copies the "winners" to a new folder which you can then train on with your favorite trainer.
   This version's defaults are tuned to create a dataset suitable for use in Fizgig using Fizgig's default-settings (For a character-LoRA of a human, not a style-LoRA)

- **face_mediapipe_bounding_box_saver**:
   - version 1.0.0
   - Reads a source-folder of assorted images of a person you want to train a LoRA on, and then looks for faces and ranks all images after the actual face-size in pixels (From largest to smallest), and then copies all the detected faces (Cropped to the actual size of the detected bounding-box) to a new folder where you can then evaluate the results manually. The file-names will be structured like this: "A_wBhCtD", where A is the rank on the hitlist (1 being the largest) so the images sort correctly using the file-name, B is the width of the bounding-box in pixels, C is the height, D is the total amount of pixels in the bounding-box (Width times height) which is used for scoring the size (This is to avoid a 50x50 image from scoring lower than a 10x100 image). A file-name example could be: "1_w930h930t864900.png", which means the largest of the batch, with a width and height of 930 pixels resulting in a total pixel-amount of 864900.
   The same detection and sorting mechanism is used in the "LoRA_Dataset_Optimizer", so the purpose of the "face_mediapipe_bounding_box_saver" is mostly so one can verify what the optimizer actually "sees".

## Prerequisites
Before installing, ensure you have the following software installed on your Windows system:

1. **Python 3.11.9**:
   - Download from [python.org](https://www.python.org/downloads/release/python-3119/).
   - Verify (With the VENV active) with: `python --version` (should output `Python 3.11.9`).
   - The individual scripts may or may not work with other versions of Python, but the .bat file expects version 3.11 (With the VENV named 'venv311') and the scripts have only been tested with version 3.11.9 (And are confirmed INcompatible with Python 3.14)

2. **System Requirements**:
   - The scripts are only tested on Windows 11.

## Installation
Follow these steps in order to set up the project:

1. **Create a Virtual Environment**:
   ```bash
   py -3.11 -m venv venv311
   ```

2. **Activate the Virtual Environment**:
   ```bash
   venv311\Scripts\activate
   ```

3. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application (Without the bat-file)
1. **Activate the Virtual Environment**:
   ```bash
   cd JEL_LoRAdatasetToolbox
   venv311\Scripts\activate
   ```

2. **Run a script**. This will open a GUI of the selected tool (This is just an example of 1 tool, so change the file-name to the desired tool if necessary) :
   ```bash
   python LoRA_Dataset_Optimizer_v1.0.0.py
   ```

## Running the Application (Using the bat-file on Windows)
1. **Double-click the bat-file** (Requires the VENV to be named "venv311") : 'LoRA_Dataset_Optimizer_v1.0.0.bat'

## License
This project is licensed under the Apache License 2.0 (modified for jurisdiction) — see the LICENSE.txt file for details. A NOTICE file is included in this repo.
