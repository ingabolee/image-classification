# Image Classification CNN

An educational and practical Deep Learning project that demonstrates the immediate architectural benefits of upgrading standard Dense Neural Networks to Convolutional Neural Networks (CNNs) for high-accuracy computer vision classification tasks.

## Overview
This repository contains a streamlined end-to-end framework leveraging TensorFlow and Keras to classify the popular `Fashion-MNIST` image dataset. The code directly compares the performance metrics of a traditional Multi-Layer Perceptron (Flatten -> Dense -> Output) against a superior Convolutional architecture (Conv2D -> MaxPooling2D -> Flatten -> Dense -> Output). This contrast effectively introduces edge detection, image compression, and localized feature extraction within deep learning models, while retaining generalization.

## Features
- **Progressive Architecture**: Implements baseline Dense models and subsequently upgrades to Spatial CNNs to demonstrate quantifiable improvements in multi-class image labeling.
- **Mathematical Conv2D Demonstrations**: Contextualizes exactly how 3x3 array kernels multiply underlying pixels dynamically across tensor matrices.
- **Max Pooling**: Combines feature detection with structural dimensionality reduction using `MaxPooling` to mitigate data explosion.
- **Custom Callbacks**: Explores TensorFlow backend callbacks stopping the training loops early once a desired threshold logic ($>99\%$ train accuracy) is fulfilled.
- **Internal Visualizations**: Iteratively plots the independent convolutional outputs over test images to identify specifically how kernels isolate edges (e.g., shoe outlines).

## Tech Stack
- Python
- TensorFlow
- Keras
- Matplotlib
- Jupyter Notebooks

## Project Architecture
```text
image-classification-master/
  image_classification.ipynb    # Interactive instructional notebook detailing convolutions
  image_classification.py       # Flat script containing the full model run environments
```

## Installation
Ensure you possess an active Python environment setup with TensorFlow dependencies for tensor processing:
```bash
pip install tensorflow matplotlib
```
*Note: Depending on the host hardware, running TensorFlow against the GPU (`tensorflow-gpu`) will substantially reduce fitting epochs times.*

## Running the Project
The application downloads the Fashion MNIST dataset automatically mapped under Keras utilities and executes locally:

**Script Execution**:
```bash
python image_classification.py
```

**Interactive Exploration**:
```bash
jupyter notebook image_classification.ipynb
```

## Model Card

### Model Overview
A deep, multi-categorical classification algorithm built to infer a subset $1$ out of $10$ designated apparel outcomes from a normalized $28 \times 28$ grayscale pixel matrix.

### Model Architecture (CNN)
- **Input Scaling**: Black and white pixels compressed from $0$ \- $255$ standard scale down to a standardized mathematical scale $0.0$ \- $1.0$ within a reshaped volume tensor $(60000, 28, 28, 1)$.
- **Conv2D / MaxPool 1**: Creates $64$ convolving feature grids running via a `$3 \times 3$` kernel combined with a `ReLU` activation metric. Followed tightly by a `$2 \times 2$` Maximum Pooling algorithm.
- **Conv2D / MaxPool 2**: Applies a secondary convolution structure capturing hierarchical components above previous edge maps.
- **Dense Output Blocks**: Flattens remaining spatial pools into a 1-dimensional array before running against a final 128 Neuron `ReLU` layer converging on a 10 Node `Softmax` array (projecting probability curves per class).

### Training Process
- Employs the `adam` sparse categorical optimizer.
- Calculates metric progression via classic Accuracy against Crossentropy mathematical loss.
- Uses dynamic `tf.keras.callbacks.Callback` hooks configured to intercept validation markers.

### Limitations
- Evaluates against relatively small low-resolution objects ($28 \times 28$). Transitioning layers directly over full-color $1080p$ arrays would be exponentially costly without deeper architectural striding.
- At epoch numbers exceeding roughly $\sim20$ runs on simplistic training features, it highlights significant deviations where Dense configurations fall drastically to **Overfitting**.

## Professional Highlights
- **Showcased model optimizations** by transitioning raw dimensional pipelines towards hardware-accelerated parameter kernels (CNNs) reducing computation footprints.
- **Implemented event-driven automation** mapping class extensions on top of Keras architectures (`myCallback`) creating early-stopping bounds.
- **Developed rich data pipelines**: Dynamically shaping dimensional bounds specifically required by backend `.fit()` endpoints (`.reshape(60000, 28, 28, 1)`).

## License
MIT License

## Contributing
Contributions are welcome. Feel free to open issues or submit pull requests for enhancements.

## Author
Lih Ingabo
