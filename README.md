
# Video-Based Human Activity Recognition Using Deep Learning

## Overview

This project implements a video-based Human Activity Recognition (HAR) pipeline combining object detection, multi-object tracking, deep visual feature extraction and temporal modeling.

The objective is to recognize human activities from video sequences by extracting person-specific visual features and learning their temporal patterns.

The proposed pipeline combines YOLOv8 for person detection, DeepSORT for tracking, ResNet50 for visual feature extraction and an LSTM network for activity classification.

## Methodology

### 1. Dataset Preparation

- Dataset: UCF101 (Kaggle version).
- Selected five activity classes:
  - WalkingWithDog
  - JumpingJack
  - LongJump
  - CliffDiving
  - BenchPress
- Group-based train/validation/test splitting to separate video groups.
- Video preprocessing and frame sampling.

### 2. Person Detection and Tracking

**YOLOv8**
- Detect people in video frames.
- Filter detections to the person class.
- Extract bounding boxes and confidence scores.

**DeepSORT**
- Track detected people across consecutive frames.
- Assign consistent tracking IDs.
- Maintain individual person trajectories.

### 3. Visual Feature Extraction

**ResNet50**
- Use pretrained ResNet50 (ImageNet weights).
- Extract person crops using tracking bounding boxes.
- Resize and normalize input images.
- Extract 2048-dimensional visual feature vectors.

The longest detected person track is selected to represent each video during feature extraction.

### 4. Temporal Modeling and Classification

**LSTM (Long Short-Term Memory)**

- Build temporal sequences from extracted visual features.
- Use sliding windows of 16 consecutive feature vectors.
- Apply an LSTM network to learn temporal activity patterns.
- Use a fully connected layer for five-class classification.
- Train using Adam optimizer and weighted cross-entropy loss.

### 5. Evaluation

Evaluate the model at two levels:

- **Window-level:** Classification of individual temporal windows.
- **Video-level:** Aggregate predictions across windows belonging to the same video.

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion matrix

The complete pipeline's processing speed is also measured.

## Architecture

```text
Input Video
    |
    v
Frame Sampling
    |
    v
YOLOv8 Person Detection
    |
    v
DeepSORT Tracking
    |
    v
Person Bounding Boxes
    |
    v
ResNet50 Feature Extraction
    |
    v
2048-D Feature Sequences
    |
    v
Sliding Window Segmentation
    |
    v
LSTM Temporal Modeling
    |
    v
Activity Classification
    |
    v
Predicted Human Activity
```

## Results

The model was evaluated on five UCF101 activity classes.

| Metric | Result |
|---|---:|
| Window-level Test Accuracy | 90.61% |
| Window-level Macro F1-score | 87.13% |
| Video-level Test Accuracy | 85.89% |
| Video-level Macro F1-score | 86.26% |
| Processing Speed | 24.4 FPS |

**Hardware:** NVIDIA Tesla T4 GPU.

The processing speed was measured on a test video, excluding the first five warm-up frames.

## Technologies

- **Programming:** Python
- **Deep Learning:** PyTorch
- **Object Detection:** YOLOv8
- **Multi-Object Tracking:** DeepSORT
- **Feature Extraction:** ResNet50
- **Temporal Modeling:** LSTM
- **Computer Vision:** OpenCV
- **Data Processing:** NumPy, Pandas
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib

## How to Run

The notebook was developed using Kaggle with GPU acceleration.

1. Open the notebook in Kaggle.
2. Add the UCF101 Action Recognition dataset.
3. Enable GPU acceleration.
4. Install the required dependencies.
5. Run the notebook cells sequentially.

The YOLOv8 and ResNet50 pretrained weights are loaded automatically.

## Limitations

- The experiments focus on five selected UCF101 activity classes.
- Videos without sufficiently detected and tracked people are excluded.
- The longest detected person track is used to represent each video.
- Performance may vary with video quality, camera viewpoints and detection failures.
