# DenseNet121-from-Scratch

An implementation of the **DenseNet-121** computer vision architecture from scratch using PyTorch.

## Introduction to the DenseNet Architecture

Deep learning has become one of the most influential fields in modern computing. Deep learning models are used to solve a wide variety of problems in areas such as computer vision, natural language processing, healthcare, finance, and robotics.

In this document, we will explore **DenseNet**, how it works, and the architectural principles behind it.

**DenseNet** stands for **Densely Connected Convolutional Networks**. It was introduced by **Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger** in the paper *Densely Connected Convolutional Networks* (CVPR 2017).

Unlike ResNet, where the output of a residual block is added to its input through a skip connection, DenseNet introduces a different connectivity pattern. Instead of residual blocks, DenseNet is built using structures known as **Dense Blocks**.

A DenseNet architecture consists of multiple Dense Blocks separated by **Transition Layers**. The number of Dense Blocks and Dense Layers determines the depth of the network.

Within each Dense Block are smaller units called **Dense Layers**.

A Dense Layer typically consists of:

* Batch Normalization
* ReLU activation
* Convolution operations

In traditional CNN architectures, each layer receives input only from the previous layer. In DenseNet, every Dense Layer receives the outputs of all preceding layers within the same Dense Block.

This dense connectivity improves information flow, encourages feature reuse, and helps alleviate the vanishing gradient problem.

Instead of adding feature maps together as in ResNet, DenseNet concatenates feature maps along the channel dimension. This allows information from earlier layers to be preserved and reused throughout the network.

![pic](https://github.com/Cohegen/Densenet121-from-Scratch/blob/main/assets/densenet.webp)

---

## Dense Connectivity Pattern

The key idea behind DenseNet is that each layer receives all previously computed feature maps as input.

For a Dense Block with several layers:

```text
x₀ → Layer₁ → x₁

x₀,x₁ → Layer₂ → x₂

x₀,x₁,x₂ → Layer₃ → x₃

...
```

Mathematically, the output of the *l*-th layer is given by:

```text
x_l = H_l([x_0, x_1, ..., x_{l-1}])
```

where:

* (H_l) represents the transformation performed by the Dense Layer.
* ([\cdot]) denotes concatenation.

This connectivity pattern is what gives DenseNet its name.

---

# Deep Dive into the Architecture

## Initial Convolutional Stem

Before entering the first Dense Block, the input image passes through an initial feature extraction stage consisting of:

1. A 7×7 convolution with 64 filters and stride 2.
2. Batch Normalization.
3. ReLU activation.
4. A 3×3 max-pooling layer with stride 2.

This stage extracts low-level visual features while reducing the spatial dimensions of the input image.

```text
Input
  │
7×7 Conv (64 channels, stride=2)
  │
BatchNorm
  │
ReLU
  │
3×3 MaxPool (stride=2)
  │
Dense Block 1
```

---

## Dense Layer

A Dense Layer is the fundamental building block of a Dense Block.

DenseNet-121 uses the **DenseNet-BC** variant, where each Dense Layer follows a bottleneck design:

1. Batch Normalization
2. ReLU
3. 1×1 Convolution (Bottleneck Layer)
4. Batch Normalization
5. ReLU
6. 3×3 Convolution

The 1×1 convolution acts as a bottleneck layer, improving computational efficiency by projecting the input features into an intermediate representation of **4k channels** before applying the 3×3 convolution. The final 3×3 convolution produces **k new feature maps**.

The implementation is shown below:

```python
class DenseLayer(nn.Module):
    """
    A Dense Layer used within a Dense Block.

    Args:
        in_shape: Number of input channels.
        growth_rate: Number of new feature maps produced.
    """

    def __init__(self, in_shape: int, growth_rate: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_shape),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_shape,
                4 * growth_rate,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                4 * growth_rate,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)
```

The output of each Dense Layer consists of only **growth_rate** new feature maps, which are then concatenated with all previously generated feature maps.

---

## Dense Block

Dense Blocks are the core computational units of DenseNet.

A Dense Block consists of multiple Dense Layers, where each layer receives feature maps from all preceding layers.

This connectivity encourages feature reuse and improves information and gradient propagation throughout the network.

The implementation is shown below:

```python
class DenseBlock(nn.Module):
    """
    A Dense Block containing multiple Dense Layers.

    Args:
        in_shape: Number of input channels.
        growth_rate: Number of new feature maps each layer produces.
        num_layers: Number of Dense Layers.
    """

    def __init__(
        self,
        in_shape: int,
        growth_rate: int = 32,
        num_layers: int = 6
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        current_in_shape = in_shape

        for _ in range(num_layers):
            self.layers.append(
                DenseLayer(
                    current_in_shape,
                    growth_rate
                )
            )

            current_in_shape += growth_rate

    def forward(self, x: torch.Tensor):
        features = [x]

        for layer in self.layers:
            new_features = layer(
                torch.cat(features, dim=1)
            )

            features.append(new_features)

        return torch.cat(features, dim=1)
```

Notice the statement:

```python
current_in_shape += growth_rate
```

Each Dense Layer contributes **growth_rate** new feature maps.

Therefore, if a Dense Block starts with (C_{in}) channels and contains (L) Dense Layers, the total number of output channels becomes:

```text
C_out = C_in + (L × k)
```

where:

* (C_{in}) = number of input channels
* (C_{out}) = number of output channels
* (L) = number of Dense Layers
* (k) = growth rate

This progressive growth in the number of channels is a defining characteristic of DenseNet.

---

## Transition Layer

Transition Layers are used to connect Dense Blocks.

They serve two primary purposes:

1. Reducing the number of feature maps.
2. Downsampling the spatial dimensions of feature maps.

Without Transition Layers, the number of channels would grow rapidly, making the network computationally expensive.

A typical Transition Layer consists of:

1. Batch Normalization
2. ReLU activation
3. 1×1 Convolution
4. Average Pooling

The implementation is shown below:

```python
class TransitionLayer(nn.Module):
    def __init__(
        self,
        in_shape: int,
        out_shape: int
    ):
        super().__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_shape),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_shape,
                out_shape,
                kernel_size=1,
                bias=False
            ),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)
```

This follows the sequence:

```text
BatchNorm → ReLU → 1×1 Convolution → Average Pooling
```

DenseNet-121 uses a compression factor:

```text
θ = 0.5
```

This means that the number of output channels after a Transition Layer is:

```text
C_out = floor(θ × C_in)
```

For DenseNet-121:

```text
C_out = floor(0.5 × C_in)
```

Thus, each Transition Layer approximately halves the number of channels before downsampling.

---

## Growth Rate (k)

The growth rate ((k)) is one of the most important hyperparameters in DenseNet.

It defines the number of new feature maps produced by each Dense Layer.

For example:

```text
Growth Rate = 32
```

means every Dense Layer contributes 32 new feature maps.

A larger growth rate increases the representational capacity of the network because more information is added at every layer. However, it also increases memory consumption and computational cost.

DenseNet-121 uses:

```text
k = 32
```

---

## DenseNet-121 Architecture

DenseNet-121 consists of four Dense Blocks separated by Transition Layers.

```text
Input Image
     │
7×7 Convolution (64 Channels)
     │
3×3 Max Pooling
     │
Dense Block 1 (6 Layers)
     │
Transition Layer
     │
Dense Block 2 (12 Layers)
     │
Transition Layer
     │
Dense Block 3 (24 Layers)
     │
Transition Layer
     │
Dense Block 4 (16 Layers)
     │
BatchNorm
     │
Global Average Pooling
     │
Fully Connected Layer
     │
Output
```

The configuration of DenseNet-121 is:

```python
block_config = [6, 12, 24, 16]
growth_rate = 32
compression = 0.5
```

---

## Why Is It Called DenseNet-121?

The number **121** refers to the total number of learnable layers in the network.

The count is obtained as follows:

| Component                          | Layers |
| ---------------------------------- | ------ |
| Initial 7×7 Convolution            | 1      |
| Dense Block 1 (6 × 2 Conv Layers)  | 12     |
| Dense Block 2 (12 × 2 Conv Layers) | 24     |
| Dense Block 3 (24 × 2 Conv Layers) | 48     |
| Dense Block 4 (16 × 2 Conv Layers) | 32     |
| Transition Layers (1 Conv Each)    | 3      |
| Final Fully Connected Layer        | 1      |

Total:

```text
1 + 12 + 24 + 48 + 32 + 3 + 1 = 121
```

This gives the architecture its name: **DenseNet-121**.

The network contains approximately **8 million parameters**, while maintaining strong performance on image classification tasks.

---

## Why DenseNet Works

DenseNet offers several advantages:

* Improved gradient flow throughout the network.
* Extensive feature reuse.
* Reduced redundancy in learned representations.
* Fewer parameters than many comparable architectures.
* Strong performance on image classification benchmarks.

By allowing each layer to directly access the outputs of all preceding layers, DenseNet promotes efficient feature propagation and learning while maintaining a relatively compact model size.

---

## Dataset

The model was trained on the **CIFAR-10** dataset, which contains:

* 60,000 color images
* 10 object classes
* 50,000 training images
* 10,000 test images

Each image has a resolution of **32×32 pixels**.

---

## Results

After training the model for **20 epochs** on CIFAR-10, the following results were obtained:

| Metric                 | Value  |
| ---------------------- | ------ |
| Best Training Accuracy | 96.69% |
| Best Test Accuracy     | 91.18% |

The model achieved strong generalization performance, surpassing 91% test accuracy while maintaining high training accuracy.

---

## Conclusion

DenseNet-121 demonstrates how dense connectivity can improve information flow, encourage feature reuse, and reduce the number of parameters required to achieve strong performance.

By concatenating feature maps from all preceding layers rather than summing them, DenseNet enables each layer to access a richer set of learned representations. This design leads to improved gradient propagation, better parameter efficiency, and competitive performance across a wide range of computer vision tasks.

  
  
