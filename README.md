
# Video-RealTime-Segment

**Video-RealTime-Segment** is a Python-based real-time video processing application that performs segmentation tasks using a pre-trained model. The application captures video from a webcam or video file, processes each frame in real-time, and segments objects or scenes within the video feed.

## Features

- Real-time video capture and segmentation using a pre-trained segmentation model.
- Bounding boxes or segment masks applied to detected objects.
- Easy integration with custom models or datasets.
- Option to run on CPU or GPU for enhanced performance.

## Technologies Used

- **Python**
- **PyTorch**: Deep learning framework used to implement and run the segmentation model.
- **OpenCV**: Used for video capture, frame processing, and display.
- **Pre-trained Segmentation Models** from `torchvision`.

## Sample Output

Here's a sample output of the segmentation task:

![Segmentation Sample](result.jpeg)

## Setup Instructions

### 1. Clone the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/aprmswra/Video-RealTime-Segment.git
cd Video-RealTime-Segment
```

### 2. Set Up the Environment

#### Using Conda

Create a new Conda environment with the required dependencies:

```bash
conda env create -f environment.yml
conda activate realtime-vid-env
```

#### Or Manually Install Dependencies

Alternatively, you can manually install the required packages:

```bash
conda create --name realtime-vid-env python=3.9
conda activate realtime-vid-env
pip install torch torchvision opencv-python
```

### 3. Run the Application

To run the real-time segmentation application, use the following command:

```bash
python real_time_segmentation.py
```

The program will open a window displaying the video feed with segmented objects. Press **'q'** to exit.

## Customization

You can customize the model used for segmentation or add new features to the application. The model can be changed by modifying the `real_time_segmentation.py` script:

```python
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
```

You can also load custom-trained models or integrate additional features like object tracking or real-time performance optimizations.

## Performance Optimizations

- Use a GPU for faster real-time segmentation. The program will automatically detect CUDA if available.
- Reduce video frame resolution for higher FPS on lower-end machines.
- Experiment with different pre-trained models or custom models for specific segmentation tasks.

## Future Enhancements

- Implement object tracking to maintain segment identities across frames.
- Add support for multiple video sources (e.g., file input, IP camera streams).
- Enable saving of processed video outputs with segmented objects.
- Integration with additional deep learning models for object recognition and tracking.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the developers of PyTorch and OpenCV for their excellent libraries.
- Inspired by real-time video processing and segmentation research.
