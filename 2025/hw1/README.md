# CS375/Psych279 Homework 1: Training Your Own Neural Network

---

## Overview

In this assignment, you will implement, train, and visualize the behavior of an AlexNet convolutional neural network (CNN) using [PyTorch](https://pytorch.org/). You will visualize the kernels of the first layer of the neural network and analyze their response patterns. You will replicate some basic findings of classical work by Hubel and Wiesel in silico by measuring orientation selectivity of several artificial neurons early in the model. Specifically, you will:

1. **Implement the AlexNet model** and understand its architecture.  
2. **Implement a training loop** capable of training the network on the [ImageNet dataset](http://www.image-net.org/).  
3. **Measure kernel responses** for various spatial frequencies and orientations of sinusoidal grating stimuli.  
4. **Visualize the learned kernels** in the first layer of your model.  

---

## Background

### AlexNet Architecture

**AlexNet** is a groundbreaking convolutional neural network introduced by Krizhevsky et al. in the paper *[“ImageNet Classification with Deep Convolutional Neural Networks”](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)*. While modern networks are deeper, AlexNet remains historically significant for demonstrating the effectiveness of CNNs on large-scale image recognition tasks.

The **original AlexNet** architecture (as introduced in the paper) consists of:

1. **Conv1**: 64 kernels of size 11×11, stride 4, followed by **ReLU**  
2. **Max Pooling**: 3×3 window, stride 2  
3. **Conv2**: 192 kernels of size 5×5, padding 2, followed by **ReLU**  
4. **Max Pooling**: 3×3 window, stride 2  
5. **Conv3**: 384 kernels of size 3×3, padding 1, followed by **ReLU**  
6. **Conv4**: 384 kernels of size 3×3, padding 1, followed by **ReLU**  
7. **Conv5**: 256 kernels of size 3×3, padding 1, followed by **ReLU**  
8. **Max Pooling**: 3×3 window, stride 2  

After these **feature extraction** layers, the spatially reduced feature maps are flattened and passed through:

9. **Fully Connected (FC6)**: 4096 units + **ReLU** + **Dropout**(p=0.5)  
10. **Fully Connected (FC7)**: 4096 units + **ReLU** + **Dropout**(p=0.5)  
11. **Fully Connected (FC8)**: 1000 units (for ImageNet classes)  

In practice, there are slightly different configurations (e.g., how channels are split for two GPUs). For this assignment, you should implement a **single-GPU** version in which the above layers are sequentially defined. Use **ReLU** after each conv or fully connected layer (except the final classification layer). Ensure your final layer outputs **num_classes** logits.

You are going to implement a version of AlexNet and train it on ImageNet. You will evaluate the model during training and observer as the accuracy of the model improves.

### Sine Grating Stimuli
We will provide you with a set of special stimuli to run through your neural network and measure specific properties that individual neurons might be strongly responding to. These stimuli consist of striped patterns with various orientation angles and frequencies.

You will run these stimul through your model during training and plot the the activations of individual kernel filters to them, in order to determine the specific orientation and frequency tuning (if any) of the kernels.

---

## Environment Setup

To ensure a clean, reproducible environment, we recommend using **micromamba** (a lightweight package manager similar to conda). Below are the steps to create a dedicated Python environment named `cs375` and install the required libraries.

(Note: if you already have conda installed on your system you can simply use that instead. Skip the micromamba installation and run all other commands with conda instead of micromamba).

1. **Install micromamba**  
   - Refer to the [micromamba documentation](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) for instructions on installing micromamba for your operating system. If you already have conda or miniconda installed, you can skip this step and replace `micromamba` below with `conda`. 

2. **Create a new environment** called `cs375` (you can change the Python version if needed):

```bash
micromamba create -n cs375 python=3.11
```

Activate the new environment:
```bash
micromamba activate cs375
```

Install PyTorch and other dependencies:

```bash
pip install torch torchvision
```

# Install additional Python libraries used in this assignment:
```bash
pip install numpy matplotlib seaborn tqdm Pillow
```

After these steps, your environment should have all the packages necessary to run the code in this assignment.

You can verify the environment by running:
```bash
python -c "import torch; import torchvision; import numpy; import matplotlib; import PIL; print('All good!')"
```

---

## Data Preparation


### 1. ImageNet Dataset

For this assignment, you will use the **ImageNet** dataset. **ImageNet** is large (~1.2 million training images, ~50k validation images) and requires substantial disk space (290GB). You can request access and download it from the official website:

- [ImageNet Homepage](http://www.image-net.org/)  
- Follow their instructions to create an account and request access to the Large Scale Visual Recognition Challenge (ILSVRC) 2012 dataset.  
- Once you have the files (`ILSVRC2012_devkit_t12.tar.gz`, `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`) place them in a directory and set it as the dataset path inside `train.py`.

**UPDATE:**
Due to storage concerns, we have prepared a 10% subset of the full ImageNet dataset which you can find [here](https://storage.cloud.google.com/cs375/imagenet-mini.zip). This should require no more than 50GB in total (including the temporary scratch space you need to unzip the files). While we still strongly encourage you to use the full dataset for a more authentic experience, we will allow submissions with this reduced dataset.

### 2. Sine Grating Images

You will also need the **sine grating images** to measure how different orientations (angles) and spatial frequencies affect the responses of your learned filters. Download these images from:

- [Sine Grating Images (OSF Link)](https://osf.io/64qv3/#:~:text=sine_grating_images_20190507) - specifically download the directory called `sine_grating_images_20190507`.

Extract them into a directory named `sine_grating_images` within your project directory. (Specifically you should have several images directory in `sine_grating_images`, not another sub directory named  `sine_grating_images_20190507`).


---


## Hyperparameters

To make training manageable, we use the following default hyperparameters and learning rate schedule:

| Hyperparameter        | Default Value | Notes                                                            |
|-----------------------|---------------|------------------------------------------------------------------|
| **Epochs**            | `30`          |                                                                  |
| **Batch size**        | `32`          | Small for compatibility; increase if hardware allows             |
| **Number of workers** | `8`           | Moderate value; increase for faster data loading if supported    |
| **Momentum**          | `0.9`         |                                                                  |
| **Weight Decay**      | `0.0`         |                                                                  |
| **Random Seed**       | `1110`        | Ensures reproducibility                                          |

We use a simplified learning rate schedule to allow quicker training:

```python
def get_lr_for_epoch(epoch: int) -> float:
    """
    Learning rate schedule:
    - Epochs  1–15: 0.01
    - Epochs 16–25: 0.001
    - Epochs 26–30: 0.0001
    """
    if epoch <= 15:
        return 0.01
    elif epoch <= 25:
        return 0.001
    else:
        return 0.0001
```

This schedule gradually lowers the learning rate during training, enabling the model to converge more effectively. Initially, a higher learning rate helps rapidly explore the solution space, while progressively smaller learning rates refine the optimization and improve final performance.



---

## Assignment Details


Below we outline the tasks you **must complete** for this assignment. Please refer to the provided code template in the file `train.py` to see exactly where each piece should be implemented. **You are required** to fill in the missing parts as specified, ensuring that your final code runs correctly and produces the requested outputs.

You should be able to train the AlexNet model at about 1 epoch per hour on a modern laptop (or significantly faster on any computer with a GPU). You should be able to train out the model for a total of 30 epochs and obtain a model with relatively strong performance. The trianing runs will take quite a bit of time so we suggest you get started early and leave your computer running overnight to train the model. 

(If your specific hardware constraints prevent you from training the model for this long come see the course staff asap and we will figure something out).


## 1. Implement the **AlexNet** Model Definition

**Task**  
- In the `AlexNet` class, implement the **architecture** of AlexNet as described in lecture and/or the original [Krizhevsky et al. paper](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf).  
- Ensure your final layer outputs **num_classes** logits (default is 1,000 for ImageNet).  

**Details**  
- Place your code in the `__init__` method of `AlexNet`.  
- Use **5 convolutional layers** (e.g., [11×11], [5×5], [3×3], etc. as outlined above), each followed by **ReLU**, with **max-pooling** after certain conv layers.  
- Implement **3 fully connected layers** with **ReLU** (and **dropout** in the first two).  
- Carefully manage strides and paddings.  
- The skeleton code has a placeholder `### TODO: Implement the AlexNet model…`; replace this with your complete implementation.

---

## 2. Implement the **Forward Pass**

**Task**  
- In the `forward` method of the `AlexNet` class, write the code that **pushes the input** through the model layers.  

**Details**  
- Pass the input through your `features` sub-network (the series of convolutional layers that embeds the image), then flatten the output and pass it through your `classifier` (the series of linear layers at the end that classifies the embedding). You do not have to implement it as these two specific sub networks, this is simply a helpful convention.
- Return the final tensor of shape `[batch_size, num_classes]`.  

---

## 3. Complete the **Missing Parts of the Training Loop**

**Task**  
- In the `main()` function, fill in the **missing** training components where indicated by comments (e.g., “### TODO: Implement the forward pass, loss computation, backward pass…”).  

**Details**  
- You will see placeholders in the training loop where you need to:  
  1. Run the **forward pass** on the batch.  
  2. Compute the **loss** using `CrossEntropyLoss`.  
  3. **Backpropagate** the gradients with `loss.backward()`.  
  4. **Update** parameters via `optimizer.step()`.  
- Make sure to accumulate the training loss correctly.  

You can find a reference implementation of a PyTorch training loop [here](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

---

## 4. Implement the **Evaluation Function**

**Task**  
- In the function `evaluate_accuracy_and_loss`, write the code that computes the **top-1 accuracy** and **average loss** on a given `DataLoader`.  

**Details**  
- Use `model.eval()` and `with torch.no_grad()` to disable gradient calculations during evaluation.  
- For each batch:  
  1. Forward pass the images.  
  2. Compute the loss.  
  3. Compute the top-1 accuracy by comparing the model’s predictions to the labels.  
- Return the overall accuracy (in percentage) and the average loss across the entire dataset. 

---

## 5. Implement the Function to **Plot the Conv1 Kernels**

**Task**  
- In `plot_conv1_kernels(model, epoch)`, visualize the weights of the **first convolutional layer**.  

**Details**  
- Use `model.features[0]` (or similar) to access the first convolution’s weights.  
- Convert them into a grid (e.g., with `torchvision.utils.make_grid`) and **plot** them with matplotlib.  
- Optionally normalize the weights so that distinct features are visible.  
- Save the figure to a file (e.g., `out/conv1_kernels_epoch_{epoch}.png`).  

---

## 6. Implement the Function to **Plot the Kernel Responses** for Sine Gratings

**Task**  
- Most of the sine grating response code (e.g., **loading images**, **parsing deg/sf**, **computing the center response in Conv1**) is already provided.  
- You only need to **(5)** create the figure for each kernel with three subplots, **(6)** compute the circular variance of each kernel and **(7)** save these plots.

**Details**  
1. We have provided the logic for loading and transforming each image from `sine_grating_images/`.  
2. We have provided a snippet showing how to parse the orientation (`deg`) and spatial frequency (`sf`) from the filename.  
3. We have provided the code that does a forward pass **only through the first conv layer** and grabs the response at `(27, 27)`.  
4. We store the results per kernel in a structure called `responses_per_kernel`.  
5. **You must create a figure** for each kernel that includes three subplots:  
   - **Response vs. Orientation**  
   - **Response vs. Spatial Frequency**  
   - **A visualization** of the kernel itself (see **plot_conv1_kernels** as a reference).  
6. **You must compute** the circular variance for each filter using the provided method and store it in a list which will be used for plotting.
7, **Save** each figure into `out/kernel_responses_{epoch}/`.  

In this section you will compute the **circular variance** of each filter’s responses **with respect to orientation** to measure how orientation-selective the filter is. We provide a helper function `compute_circular_variance`, but you should still understand the math behind it. In short, the **circular variance** (CV) is defined:

$$
\text{CV} = 1 - \frac{
   \sqrt{\Bigl(\sum_i r_i \cos(\theta_i)\Bigr)^2 + \Bigl(\sum_i r_i \sin(\theta_i)\Bigr)^2}
}{
   \sum_i r_i
}
$$

where:
- $(r_i \ge 0)$ is the neuron’s response to orientation $(\theta_i)$ (in radians).
- $(\sum_i r_i \cos(\theta_i))$ and $(\sum_i r_i \sin(\theta_i))$ are the *x*- and *y*-components of the resultant vector, respectively.
- $(\sum_i r_i)$ is the total response (i.e., sum of all $(r_i)$ values).

A **CV** near **0** indicates that the neuron is highly *directional* (i.e., has a strong preference for one orientation), while a **CV** near **1** indicates a *spread-out* response across orientations (i.e., no clear preference).

Your code should compute the CV for each **kernel** in the first convolutional layer and add it to the provided list. This will help demonstrate how some kernels become highly selective for particular orientations while others are more broadly tuned.


---

## Putting It All Together

When you **run** your script, the following steps should occur:

1. **Data Loading**: ImageNet (or a subset) is loaded into `train_loader` and `test_loader`.  
2. **Model Initialization**: `AlexNet` is instantiated and moved to the appropriate device.  
3. **Training**: For each epoch, you update the learning rate if needed, then run the training loop and record training loss.  
4. **Evaluation**: After each epoch, you run `evaluate_accuracy_and_loss` to get test/validation metrics.  
5. **Plotting**:  
   - **Training Metrics** (already provided in the skeleton): A figure of train/test loss and test accuracy over epochs.  
   - **Conv1 Kernels**: `plot_conv1_kernels(model, epoch)` saves a grid of first-layer filters.  
   - **Kernel Responses**: `plot_sine_grating_responses_for_filters(model, epoch, device)` saves orientation/frequency response plots for each kernel and a histogram of the circular variance distribution across filters.
6. **Checkpointing**: The code saves `model.pt` so you can resume training if needed.

Please make sure to **fill in** all areas marked with `### TODO` in the provided code template. When complete, your script should run end-to-end and produce the **outputs** described above in the `out/` directory.

---

## Submission Instructions

1. **Code**
   - [10 Points] Submit your modified `train.py` file with all tasks completed. Include your name at the top of the assigment.

2. **Report**
   - Provide a PDF or Markdown report that includes:  
     - [2 Points] A brief explanation of the code you implemented
     - [3 Points] An image of the accuracy, loss and circular variance plot, along with a description of the final accuracy values. Observe the trends in loss decrease, accuracy increase and kernel circular variance. Specifically remark on when during the training do the filters seem to get tuned for direction selectivity.
     - [2 Points] An image of the kernel visualization plot of the first layer along with a brief description of some qualitative properties of some of the filters.
     - [3 Points] Visualizations of 3 individual filters of your choice and their rotation and frequency selectivity plots. Pick filters that illustrate a clear bias and describe what they seem to be selective for.

You will submit both files on the submission link on Canvas.

You are allowed to use any tools you wish (including LLMs) to help with this assigment, but you must work by yourself. If you encounter any issues with setting up the training on your computer please contact the course staff sooner rather than later.

---

**Good luck with your implementation!** If you have any questions, please reach out via email to klemenk@stanford.edu.
