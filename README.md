# Vehicle Detection using Convolutional Neural Network
# version 0.0.0.1

This project utilizes a Convolutional Neural Network (CNN) to detect vehicles in images. The model is trained using the Caffe deep learning framework.

## Project Structure

- `deploy.prototxt`: The provided deploy.prototxt defines a MobileNet-SSD architecture for object detection. 
- `mobilenet_iter_730000.caffemodel`: The MobileNet-SSD contains `data` with a shape of `1x3x300x300`, which means batches of 1 image, 3 channels (RGB), and 300x300 resolution.
- `detection_run.py`: Conatains the source code to run upon the model.
- `README.md`: Project documentation.

## Requirements

- Python 3.x
- NumPy
- OpenCV
- playsound module
- OS
- threading
- time 
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Muhamid786/Vehicle_Detection_Caffemodel.git
    ```
2. Install the required Python packages:
    ```bash
    using `pip install -r <package_name>` command if any missing.
    ```

## Usage
To test the project, run:
```bash
python3 detection_run.py
```

### Training and Testing
The project uses a pretrained MobileNet-SDD architecture, hence it doesn't need to be trained and tested further.

## Acknowledgements

- The Caffe team for their deep learning framework.
- Any other contributors or resources.
