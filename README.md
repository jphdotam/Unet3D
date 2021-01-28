# Unet3D
A 3D Unet for Pytorch for video and 3D model segmentation

This is a 3D model I adapted and optimized for 3D from the 2D Unet at https://github.com/milesial/Pytorch-UNet.

I've used this for 3D segmentation and also pose detection (with MSEloss) tasks with surprising success.

To give an idea of what you can achieve, I can use a batch size of around 10 on a 32 GB GPU with input videos of 32 frames of 256 * 192 pixels in grayscale, at full precision.

This model appears to work MUCH better than other approaches such as the similar VNet model here: https://github.com/mattmacy/vnet.pytorch

Hopefully the usage is straight forward.

`model = n_channels, n_classes, width_multiplier=1, trilinear=True, use_ds_conv=False)`

Where:

* `n_channels` is the depth of the input data (1 for grayscale input videos, 3 for RGB)

* `n_classes` is the number of output channels (e.g. classes for segmentation)

* `width_multiplier` allows the number of filters to be increases/decreased linearly, from the default of (32, 64, 128, 256, 512)

* `trilinear` means trilinear interpolation is used for the upsampling, rather the ConvTranspose layers. This means fewer parameters for the model. I haven't extensively tested which works best, so it's worth experimenting.

* `use_ds_conv` allows depthwise-separable convolutions to be used; I find this saves relatively little VRAM I think is of, frankly, limited utility
