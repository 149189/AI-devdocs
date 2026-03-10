// 08 - Computer Vision Advanced
(function () {
  const content = {
    image_preprocessing: `# Image Preprocessing

Image preprocessing transforms raw images into suitable input for computer vision models. Proper preprocessing directly impacts model performance and training efficiency.

## Key Techniques

| Technique | Description | Purpose |
|-----------|-------------|---------|
| Resizing | Scale to fixed dimensions | Uniform input size for batching |
| Normalization | Scale pixel values to [0,1] or [-1,1] | Faster convergence |
| Standardization | Zero mean, unit variance per channel | Match pretrained model expectations |
| Augmentation | Random transforms | Increase data diversity |
| Color space conversion | RGB to grayscale/HSV/LAB | Task-specific representations |

## Data Augmentation

\`\`\`python
import torchvision.transforms as T

transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])  # ImageNet stats
])
\`\`\`

## Advanced Augmentation

- **Cutout**: Randomly mask square regions
- **Mixup**: Blend two images and their labels
- **CutMix**: Replace patch of one image with another
- **AutoAugment**: Learned augmentation policies
- **RandAugment**: Simplified random augmentation

## Applications

- Medical imaging (contrast enhancement, artifact removal)
- Satellite imagery (atmospheric correction, pansharpening)
- Autonomous driving (weather normalization)
- Document scanning (deskewing, binarization)

## Evolution

- **1960s**: Histogram equalization for contrast enhancement
- **2012**: Data augmentation becomes essential in AlexNet training
- **2017**: Cutout and Mixup introduce creative augmentation
- **2019**: AutoAugment learns optimal augmentation policies
- **2020s**: Self-supervised augmentation strategies (SimCLR, BYOL)`,

    feature_extraction: `# Feature Extraction

Feature Extraction identifies and represents distinctive patterns in images. It bridges raw pixels and higher-level understanding needed for recognition and matching tasks.

## Traditional Features

| Feature | Type | Use Case |
|---------|------|----------|
| SIFT | Keypoint | Scale-invariant matching |
| SURF | Keypoint | Fast approximate SIFT |
| HOG | Gradient histogram | Pedestrian detection |
| LBP | Texture | Face recognition |
| Harris Corners | Corner detection | Image alignment |
| ORB | Keypoint | Real-time matching |

## How Traditional Features Work

\`\`\`python
import cv2

# SIFT Feature Detection
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

# HOG (Histogram of Oriented Gradients)
from skimage.feature import hog
features, hog_image = hog(image, orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          visualize=True)
\`\`\`

## Deep Learning Features

Modern feature extraction uses pretrained CNN/ViT models as feature extractors:

- **Transfer Learning**: Use pretrained model without final classification layer
- **Feature Pyramid Networks (FPN)**: Multi-scale feature extraction
- **Vision Transformers (ViT)**: Patch-based self-attention features
- **CLIP Features**: Joint vision-language representations

## Applications

- Image matching and stitching (panoramas)
- Visual search and similarity
- Place recognition in robotics
- Content-based image retrieval
- Transfer learning backbone selection

## Evolution

- **1999**: SIFT introduced by Lowe (scale-invariant features)
- **2005**: HOG for pedestrian detection (Dalal & Triggs)
- **2012**: AlexNet shows learned features outperform handcrafted
- **2020**: Vision Transformers extract patch-based features
- **2021**: CLIP learns universal vision-language features`,

    object_tracking: `# Object Tracking

Object Tracking follows specific objects across video frames, maintaining identity and trajectory over time. It is essential for surveillance, autonomous driving, and sports analytics.

## Types

| Type | Description | Example |
|------|-------------|---------|
| Single Object (SOT) | Track one target | Player tracking in sports |
| Multi-Object (MOT) | Track many targets simultaneously | Pedestrian tracking |
| Online | Process frames sequentially | Real-time applications |
| Offline | Process all frames together | Post-processing video |

## Key Concepts

- **Detection**: Finding objects in each frame
- **Association**: Linking detections across frames to maintain identity
- **Re-identification (ReID)**: Recognizing same object after occlusion
- **Kalman Filter**: Predicting object motion for smoother tracking
- **IoU (Intersection over Union)**: Measuring detection overlap

## How It Works

\`\`\`
Tracking-by-Detection Pipeline:
1. Detect objects in each frame (YOLO, Faster R-CNN)
2. Predict position using motion model (Kalman filter)
3. Associate predictions with detections (Hungarian algorithm)
4. Handle births (new objects) and deaths (lost objects)

Key Algorithms:
- SORT: Simple Online Realtime Tracking (Kalman + Hungarian)
- DeepSORT: SORT + deep appearance features for ReID
- ByteTrack: Associates all detection boxes (high + low confidence)
- BoT-SORT: Combines motion and appearance cues
\`\`\`

## Applications

- Autonomous driving (tracking vehicles and pedestrians)
- Sports analytics (player tracking, ball trajectory)
- Surveillance and security
- Retail analytics (customer flow)
- Wildlife monitoring

## Evolution

- **2006**: Particle filters for visual tracking
- **2016**: SORT introduces simple real-time multi-object tracking
- **2017**: DeepSORT adds appearance features
- **2022**: ByteTrack achieves state-of-the-art with simple approach
- **2024+**: Transformer-based trackers and tracking-anything models (SAM 2)`,

    vision_3d: `# 3D Vision & Depth Estimation

3D Vision reconstructs three-dimensional understanding of scenes from 2D images. Depth estimation predicts the distance of each pixel from the camera.

## Key Concepts

- **Depth Map**: Per-pixel distance from camera
- **Point Cloud**: Set of 3D points representing a scene
- **Stereo Vision**: Using two cameras to compute depth via disparity
- **Structure from Motion (SfM)**: Reconstructing 3D from multiple views
- **NeRF**: Neural Radiance Fields for novel view synthesis
- **Monocular Depth**: Estimating depth from a single image

## Methods

| Method | Input | Output |
|--------|-------|--------|
| Stereo Matching | Stereo image pair | Dense depth map |
| Monocular Depth | Single image | Relative depth map |
| LiDAR | Laser scanning | Precise point cloud |
| Structure from Motion | Multiple views | 3D reconstruction |
| NeRF | Multiple views + poses | Neural 3D scene |
| Gaussian Splatting | Multiple views | Real-time 3D rendering |

## How It Works

\`\`\`python
# Monocular depth estimation with MiDaS
import torch
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
# Input: single RGB image
# Output: relative depth map

# NeRF (Neural Radiance Fields)
# Train: Optimize MLP to predict color + density for 3D points
# Query: For any 3D point and viewing direction,
#         predict (r, g, b, density)
# Render: Ray marching through the neural field
\`\`\`

## Applications

- Autonomous driving (3D object detection, obstacle avoidance)
- Augmented reality (placing virtual objects in real scenes)
- Robotics (grasp planning, navigation)
- Architecture and construction (3D scanning)
- Virtual reality content creation

## Evolution

- **1980s**: Stereo vision and SfM algorithms developed
- **2014**: Depth estimation with CNNs (Eigen et al.)
- **2020**: NeRF introduces neural 3D scene representation
- **2022**: MiDaS achieves robust monocular depth estimation
- **2023**: 3D Gaussian Splatting enables real-time neural rendering`,

    video_understanding: `# Video Understanding

Video Understanding analyzes temporal visual data to recognize actions, events, and narratives. It extends image understanding with the dimension of time.

## Key Tasks

| Task | Description | Example |
|------|-------------|---------|
| Action Recognition | Classify actions in clips | Walking, running, cooking |
| Temporal Detection | Locate actions in time | When does the goal happen? |
| Video Captioning | Generate text descriptions | "A person is riding a bicycle" |
| Video QA | Answer questions about video | "What color is the car?" |
| Video Summarization | Create shorter highlight reels | Sports highlights |

## Key Concepts

- **Temporal Modeling**: Understanding how frames relate over time
- **Optical Flow**: Pixel-level motion between frames
- **3D Convolutions**: CNNs that convolve across space AND time
- **Two-Stream Networks**: Separate processing for appearance and motion
- **Video Transformers**: Self-attention across spatial and temporal dimensions

## Architectures

\`\`\`
Evolution of Video Models:
1. Two-Stream (2014): RGB + Optical Flow -> fuse predictions
2. C3D / I3D (2015-2017): 3D convolutions on video volumes
3. SlowFast (2019): Slow (spatial) + Fast (temporal) pathways
4. TimeSformer (2021): Divided space-time attention
5. VideoMAE (2022): Self-supervised video pre-training
6. Video LLMs (2024): GPT-4V, Gemini for video understanding
\`\`\`

## Applications

- Content moderation (detecting violent or inappropriate content)
- Sports analytics (play recognition, strategy analysis)
- Security and surveillance (anomaly detection)
- Medical video analysis (surgical procedure recognition)
- Autonomous driving (predicting pedestrian behavior)

## Evolution

- **2014**: Two-stream networks for video recognition
- **2017**: I3D inflates 2D ImageNet features to 3D
- **2019**: SlowFast networks for efficient video understanding
- **2022**: VideoMAE enables self-supervised video pretraining
- **2024+**: Multimodal LLMs process video directly (GPT-4o, Gemini)`,

    ocr: `# OCR & Document AI

Optical Character Recognition (OCR) converts images of text into machine-readable characters. Document AI extends this to understand document structure, layout, and semantics.

## Key Concepts

- **Character Recognition**: Identifying individual characters/glyphs
- **Text Detection**: Locating text regions in images
- **Text Recognition**: Reading detected text regions
- **Layout Analysis**: Understanding document structure (headers, paragraphs, tables)
- **End-to-End**: Detecting and recognizing text in a single model

## OCR Pipeline

\`\`\`
Traditional OCR Pipeline:
1. Image preprocessing (binarization, deskewing)
2. Text detection (locate text regions)
3. Text recognition (read characters in regions)
4. Post-processing (spell check, formatting)

Modern End-to-End:
1. Input image
2. Deep learning model (CRNN, TrOCR, PaddleOCR)
3. Structured output with text + positions
\`\`\`

## Key Tools

| Tool | Type | Strength |
|------|------|----------|
| Tesseract | Open-source OCR | 100+ languages, free |
| PaddleOCR | Deep learning OCR | Fast, accurate, multilingual |
| TrOCR | Transformer OCR | State-of-the-art accuracy |
| EasyOCR | Python library | Easy to use, 80+ languages |
| Azure Document Intelligence | Cloud API | Enterprise-grade document AI |
| Google Document AI | Cloud API | Layout and entity extraction |

## Applications

- Digitizing printed books and archives
- Invoice and receipt processing
- License plate recognition
- Form data extraction
- Handwriting recognition
- Accessibility (screen readers for images)

## Evolution

- **1950s**: First OCR machines for postal sorting
- **1985**: Tesseract OCR engine created by HP
- **2006**: Tesseract open-sourced by Google
- **2015**: Deep learning OCR (CRNN) surpasses traditional methods
- **2021**: TrOCR (Transformer-based) sets new benchmarks
- **2024+**: Vision LLMs handle OCR as part of general understanding`,

    medical_imaging: `# Medical Imaging AI

Medical Imaging AI applies computer vision and deep learning to analyze medical images for diagnosis, treatment planning, and research. It augments radiologists and clinicians with automated detection and analysis.

## Key Modalities

| Modality | Imaging Type | Common AI Tasks |
|----------|-------------|----------------|
| X-ray | 2D projection | Pneumonia, fracture detection |
| CT Scan | 3D volumetric | Tumor detection, lung nodules |
| MRI | 3D soft tissue | Brain tumors, cardiac analysis |
| Ultrasound | Real-time | Fetal monitoring, echocardiography |
| Pathology | Microscopy slides | Cancer grading, cell counting |
| Retinal | Fundus photos | Diabetic retinopathy screening |

## Key Tasks

- **Classification**: Is this X-ray normal or abnormal?
- **Detection**: Locate tumors, nodules, or fractures
- **Segmentation**: Outline organ boundaries and lesion regions
- **Registration**: Align images from different time points
- **Reconstruction**: Generate high-quality images from sparse data

## How It Works

\`\`\`
Medical Imaging AI Pipeline:
1. Image Acquisition: DICOM format from scanners
2. Preprocessing: Windowing, normalization, resampling
3. Model Inference: CNN/ViT trained on medical data
4. Post-processing: Threshold, morphology, connected components
5. Clinical Report: Findings with confidence scores

Key architectures:
- U-Net: Gold standard for medical image segmentation
- DenseNet: Effective for classification tasks
- nnU-Net: Self-configuring U-Net for any segmentation task
- SAM (Segment Anything): Foundation model adapted for medical images
\`\`\`

## Applications

- Breast cancer detection in mammograms
- Diabetic retinopathy screening from retinal images
- COVID-19 detection from chest X-rays and CT
- Brain tumor segmentation for surgical planning
- Drug discovery through cellular imaging

## Evolution

- **2012**: Deep learning applied to medical images
- **2015**: U-Net architecture designed specifically for biomedical segmentation
- **2017**: CheXNet matches radiologist performance on chest X-rays
- **2020**: AI aids COVID-19 diagnosis from CT scans at scale
- **2024+**: Foundation models (BiomedCLIP, MedSAM) enable generalist medical AI`,

    autonomous_vision: `# Autonomous Driving Vision

Autonomous driving vision systems perceive the driving environment using cameras, LiDAR, and radar to enable self-driving vehicles to navigate safely.

## Key Tasks

| Task | Description | Sensors |
|------|-------------|---------|
| 2D Object Detection | Detect vehicles, pedestrians, signs | Camera |
| 3D Object Detection | Detect with distance and 3D bounding boxes | Camera + LiDAR |
| Semantic Segmentation | Label every pixel (road, sidewalk, car) | Camera |
| Lane Detection | Find lane markings and boundaries | Camera |
| Depth Estimation | Predict distance to objects | Camera/Stereo/LiDAR |
| Sensor Fusion | Combine multiple sensor inputs | All sensors |

## Perception Stack

\`\`\`
Autonomous Driving Perception Pipeline:
1. Sensor Input: Cameras (360deg), LiDAR, radar, IMU
2. Sensor Fusion: Combine modalities into unified representation
3. Detection: 3D bounding boxes for all objects
4. Tracking: Maintain object identity across frames
5. Prediction: Forecast other agents' future trajectories
6. Planning: Decide vehicle's actions
7. Control: Execute steering, throttle, brake commands

Key Models:
- BEVFormer: Bird's-eye view representation from cameras
- PointPillars: 3D detection from LiDAR point clouds
- CenterPoint: 3D object detection and tracking
- UniAD: Unified autonomous driving framework
\`\`\`

## Approaches

| Company | Strategy | Sensors |
|---------|----------|---------|
| Waymo | Full sensor suite | Camera + LiDAR + Radar |
| Tesla | Vision-only | Cameras only (8 cameras) |
| Cruise | Full sensor suite | Camera + LiDAR + Radar |
| Mobileye | Camera-first | Cameras + radar + mapping |

## Safety Levels (SAE)

- **Level 0-1**: No automation / Driver assistance (lane keeping)
- **Level 2**: Partial automation (adaptive cruise + lane centering)
- **Level 3**: Conditional automation (hands-off in specific conditions)
- **Level 4**: High automation (self-driving in defined areas)
- **Level 5**: Full automation (no human intervention needed)

## Evolution

- **2004**: DARPA Grand Challenge (first autonomous vehicle race)
- **2009**: Google Self-Driving Car project begins (now Waymo)
- **2016**: Tesla Autopilot with neural network-based vision
- **2020**: Waymo launches commercial robotaxi service
- **2024+**: Level 4 deployments expand; vision-language models for driving`,
  };

  Object.assign(window.AI_DOCS, content);
})();
