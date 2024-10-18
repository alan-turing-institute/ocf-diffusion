from diffusers import UNet2DModel
from cloudcasting.constants import NUM_CHANNELS
import torch


class ConditionedUnet(torch.nn.Module):
    history_steps: int
    def __init__(self, image_size, history_steps = 1):
        super().__init__()

        self.history_steps = history_steps

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (previous timesteps)
        self.model = UNet2DModel(
            sample_size=image_size,  # the target image resolution
            in_channels=NUM_CHANNELS + history_steps * NUM_CHANNELS,  # noise input + conditioning information
            out_channels=NUM_CHANNELS,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(
                128,
                256,
                512,
            ),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",  # a regular ResNet downsampling block

                # "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                # "AttnDownBlock2D",
            ),
            up_block_types=(
                # "AttnUpBlock2D",
                # "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
                "UpBlock2D",  # a regular ResNet upsampling block
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
            add_attention = False,  # blows up memory with attention -- maybe do latent diffusion
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, noisy_image, conditioning, t):

        # stack noisy image and conditioning info along the time axis
        stacked = torch.cat([noisy_image, conditioning], dim=-3)

        # reshape to (batch, channels*time, height, width)
        net_input = stacked.reshape(-1, stacked.shape[-4] * stacked.shape[-3], *stacked.shape[-2:])

        return self.model(net_input, t).sample.reshape(noisy_image.shape)