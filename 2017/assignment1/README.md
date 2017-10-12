## TODO

### Part 1
- Implement 1-Stream Variant of AlexNet (tfutils.model.alexnet ?)
- Implement one or more smaller variants of deep neural network, with fewer layers and fewer filters per layer
- Implement Inception v3 or VGG
- Edit `train_imagenet.py` (basically configuration)

AlexNet Reqs:

| Requirement | Satisfied?|
|-----|-----|
|the net contains eight layers with weights; the first five are convolutional and the remaining three are fully- connected | Yes |
| output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels. | No |
| The kernels of the second, fourth, and fifth convolutional layers are connected only to those kernel maps in the previous layer which reside on the same GPU (see Figure 2). | Not sure we need this... |
| The kernels of the third convolutional layer are connected to all kernel maps in the second layer. | Not sure if this applies ...| 
|The neurons in the fully- connected layers are connected to all neurons in the previous layer. | Yes | 
|Response-normalization layers follow the first and second convolutional layers | No (But apparently this is optional) |
| Max-pooling layers, of the kind described in Section 3.4, follow both response-normalization layers as well as the fifth convolutional layer | Yes | 
| The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3 with a stride of 4 pixels | Yes | 
| The second convolutional layer takes as input the (response-normalized and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5×5×48. | Yes (we use a stride of 1, the paper says nothing on how many strides).| 
| The third, fourth, and fifth convolutional layers are connected to one another without any intervening pooling or normalization layers. | Yes |
| The third convolutional layer has 384 kernels of size 3 × 3 × 256 connected to the (normalized, pooled) outputs of the second convolutional layer | Yes (no mention of strides)|
| The fourth convolutional layer has 384 kernels of size 3 × 3 × 192 | Yes |
| the fifth convolutional layer has 256 kernels of size 3×3×192 | Yes |
| The fully-connected layers have 4096 neurons each. | Yes | 