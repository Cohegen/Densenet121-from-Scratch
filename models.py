import torch
import torch.nn as nn


class DenseLayer(nn.Module):
  """
  A dense layer class which is used instantiate a Dense layer object
  within our dense block.

  
  """
  def __init__(self,in_shape:int,growth_rate:int):
    super().__init__()
    self.layer = nn.Sequential(
        nn.BatchNorm2d(in_shape),
        nn.ReLU(),
        nn.Conv2d(in_shape,4*growth_rate,kernel_size=1,bias=False),
        nn.BatchNorm2d(4*growth_rate),
        nn.ReLU(),
        nn.Conv2d(4*growth_rate,growth_rate,kernel_size=3,padding=1,bias=False)
    )

  def forward(self,x:torch.Tensor):
    return self.layer(x)

class DenseBlock(nn.Module):
  """
  A denseblock class who within it several dense layers are place,
  depending on the number of layers.

  Args:
      in_shape: number of channels a block's input has
      growth_rate: number of new feature maps each layer will add
      num_layers: number of layers each DenseBlock is likely to have
  """
  def __init__(self,in_shape:int,growth_rate:int=32,num_layers:int=6):
    super().__init__()

    ## A list which stores the layers
    self.layers = nn.ModuleList()

    current_in_shape =in_shape
    for _ in range(num_layers):
      self.layers.append(
          DenseLayer(current_in_shape,growth_rate)
      )
      current_in_shape += growth_rate

  def forward(self,x:torch.Tensor):
    features = [x]

    for layer in self.layers:
      new_features = layer(
          torch.cat(features,dim=1)
      )
      features.append(new_features)

    return torch.cat(features,dim=1)

class TransitionLayer(nn.Module):
  def __init__(self,in_shape:int,out_shape:int):
    super().__init__()
    self.layer = nn.Sequential(
        nn.BatchNorm2d(in_shape),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_shape,out_shape,kernel_size=1,bias=False),
        nn.AvgPool2d(kernel_size=2,stride=2)

    )

  def forward(self,x:torch.Tensor):
    return self.layer(x)

class Densenet121(nn.Module):
  def __init__(self,num_classes:int=10):
    super().__init__()
    self.stem = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1), # Less aggressive for 32x32 images
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # Removed MaxPool2d to preserve spatial dimensions
    )

    self.block1 = DenseBlock(64,32,6)
    self.transition1 = TransitionLayer(256,128)

    self.block2 = DenseBlock(128,32,12)
    self.transition2 = TransitionLayer(512,256)

    self.block3 = DenseBlock(256,32,24)
    self.transition3 = TransitionLayer(1024,512)

    self.block4 = DenseBlock(512,32,16)

    self.classification_head = nn.Sequential(
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(1024,num_classes)
    )

  def forward(self,x:torch.Tensor):
    x = self.stem(x)

    x = self.block1(x)
    x = self.transition1(x)

    x = self.block2(x)
    x = self.transition2(x)

    x = self.block3(x)
    x = self.transition3(x)

    x = self.block4(x)

    x = self.classification_head(x)

    return x
