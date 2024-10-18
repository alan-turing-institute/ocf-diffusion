import torch
import tqdm
import numpy as np
from cloudcasting.constants import NUM_FORECAST_STEPS, IMAGE_SIZE_TUPLE
from cloudcasting.models import AbstractModel
from diffusers import DDPMScheduler

import diffusion



# We define a new class that inherits from AbstractModel
class DiffusionModel(AbstractModel):
    """DiffusionModel model class"""

    def __init__(self, history_steps: int, state_dict_path: str, num_train_timesteps: int) -> None:
        # All models must include `history_steps` as a parameter. This is the number of previous
        # frames that the model uses to makes its predictions. This should not be more than 25, i.e.
        # 6 hours (inclusive of end points) of 15 minutely data.
        # The history_steps parameter should be specified in `validate_config.yml`, along with
        # any other parameters (replace `example_parameter` with as many other parameters as you need to initialize your model, and also add them to `validate_config.yml` under `model: params`)
        super().__init__(history_steps)


        ###### YOUR CODE HERE ######
        # Here you can add any other parameters that you need to initialize your model
        # You might load your trained ML model or set up an optical flow method here.
        # You can also access any code from src/diffusion, e.g.

        # Get the appropriate PyTorch device
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        # Calculate required crop for the input dims to be divisible by 16
        self.cropped_image_size = [(size // 16) * 16 for size in IMAGE_SIZE_TUPLE]

        # Load the pretrained model
        self.model = diffusion.ConditionedUnet(image_size=self.cropped_image_size)
        self.model.load_state_dict(torch.load(state_dict_path, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2"
        )
        ############################

    def crop(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :self.cropped_image_size[0], :self.cropped_image_size[1]]

    def forward(self, X):
        # This is where you will make predictions with your model
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)

        # Load data onto the GPU, crop to size and transform range to (-1, 1)
        X_torch = self.crop(torch.from_numpy(X).to(self.device) * 2 - 1)

        tensors = [X_torch]
        for forecast_timestep in range(NUM_FORECAST_STEPS):
            print(f"Starting denoising loop for forecast {forecast_timestep + 1} / {NUM_FORECAST_STEPS}")
            y_hat_torch = self.forecast_one_timestep(tensors[-1])
            tensors.append(y_hat_torch)

        # Convert results to the expected output format by doing the following:
        # - convert back to the range (0, 1)
        # - convert from Tensor to numpy array
        # - drop the input from the list of results
        # - concatenate the forecasts along the time axis
        results = [((y_hat + 1) / 2).detach().cpu().numpy() for y_hat in tensors[1:]]
        output = np.concatenate(results, axis=2)
        print(f"Returning {output.shape} {type(output)}")
        return output


    def forecast_one_timestep(self, X: torch.Tensor) -> torch.Tensor:
        print(f"-> Input: {X.shape} {type(X)}")

        # Random noise with shape (batch_size, channels, 1, height, width)
        sampled_noise = torch.randn(X.shape[0], X.shape[1], 1, X.shape[3], X.shape[4]).to(self.device)
        print(f"-> Noise: {sampled_noise.shape} {type(sampled_noise)}")

        # Sampling loop
        for idx, t in enumerate(tqdm.tqdm(self.noise_scheduler.timesteps)):

            # Get model pred
            with torch.no_grad():
                residual = self.model(sampled_noise, X, t)
            print(f"... obtained residual for step {idx} / {self.noise_scheduler.timesteps.shape[0]}")

            # Update sample with step
            sampled_noise = self.noise_scheduler.step(residual, t, sampled_noise).prev_sample
            print(f"... updated sampled noise for step {idx} / {self.noise_scheduler.timesteps.shape[0]}")

        return sampled_noise



    def hyperparameters_dict(self):

        # This function should return a dictionary of hyperparameters for the model
        # This is just for your own reference and will be saved with the model scores to wandb

        ###### YOUR CODE HERE ######
        return {
            "num_train_timesteps": self.noise_scheduler.timesteps,
        }
