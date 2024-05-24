# Dartboard Detection System

## Overview

This repository contains the implementation of a dartboard detection system utilizing a combination of Viola-Jones object detection and advanced shape detection techniques. The project focuses on accurately identifying dartboard locations in images using cascaded classifiers and geometric shape analysis.

## Project Structure

- **Dartboard/**
  - Contains essential data and scripts for the dartboard detection.
- **dartboard_detector.py**
  - Main Python script for dartboard detection using Viola-Jones and shape detectors.
- **Dartboardcascade/**
  - Directory containing the trained model files for the cascade classifier.
- **groundTruth.txt**
  - Text file containing the ground truth data for the dartboards in the images.
- **negatives/**
  - Folder with images used as negative samples during the training of the classifier.
- **negatives.dat**
  - Data file listing the negative samples.
- **report.pdf**
  - Detailed report documenting the methodology, results, and analysis of the dartboard detection system.
- **template.jpg**
  - Template image used for feature matching in the detection process.

## Detection Pipeline

1. **Viola-Jones Cascade Classifier**: Initial dartboard detection using Haar features to filter potential dartboard regions.
2. **Shape Detection**:
   - Line and circle detection with Hough Transform to confirm dartboard features.
   - Integration of detected shapes to refine dartboard localization.
3. **Feature Matching**:
   - Utilizes FLANN based matcher for robust feature matching, overcoming issues related to object orientation and scale.
   - Combination of feature matching points and geometric shapes to finalize dartboard detection.

## Highlights of Implementation

- **Reduction of False Positives**: Significant decrease in false positives through multi-stage filtering.
- **Integration with Shape Detectors**: Enhances detection accuracy by combining Viola-Jones detections with line and circle indicators from Hough Transform.
- **Improvement Strategies**:
  - Shift from template matching to feature matching to accommodate various dartboard sizes and orientations.
  - Weighted scoring system for potential dartboard regions to optimize detection accuracy.

## Running the Detector

To run the dartboard detection system, execute the `dartboard_detector.py` script:

```bash
python dartboard_detector.py
```