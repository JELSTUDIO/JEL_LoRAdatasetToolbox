# JEL_LoRAdatasetToolbox

A collection of various tools related to creating or pruning or working with data-sets needed for creating LoRA models.

## Available tools
List of currently available tools (Which may contain bugs or errors I haven't discovered yet. I use AI to assist with the coding and bug-hunting):

- **PruneSourceImageDataSetToGetCleanImageDataSetForLoraTraining**:
   - version 1.0.0
   - Reads a source-folder of assorted images of a person you want to train a LoRA on, and then looks for suitable faces and ranks all images after sharpness, and then copies the "winners" to a new folder which you can then train on with your favorite trainer.
   This version's defaults are tuned to create a dataset suitable for use in Fizgig using Fizgig's default-settings (For a character-LoRA of a human, not a style-LoRA)

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
   python PruneSourceImageDataSetToGetCleanImageDataSetForLoraTraining.py
   ```

## Running the Application (Using the bat-file on Windows)
1. **Double-click the bat-file** (Requires the VENV to be named "venv311") : 'PruneSourceImageDataSetToGetCleanImageDataSetForLoraTraining.bat'

## License
This project is licensed under the Apache License 2.0 (modified for jurisdiction) — see the LICENSE.txt file for details. A NOTICE file is included in this repo.
