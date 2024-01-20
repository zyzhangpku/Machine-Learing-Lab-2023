import torch
from VarAutoencoder import VarAutoencoder, VAE_ENCODING_DIM
from Autoencoder import Autoencoder, AE_ENCODING_DIM
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="AE", choices=["VAE", "AE"])
args = parser.parse_args()

model_save_path = f"./model/Best_{args.model}.pth"
vis_root = "./vis"
model_class = VarAutoencoder if args.model == "VAE" else Autoencoder
ENCODING_DIM = AE_ENCODING_DIM if args.model == "AE" else VAE_ENCODING_DIM

model = model_class(
    encoding_dim=ENCODING_DIM
)
model.load_state_dict(torch.load(model_save_path))
model.eval()
latent_vectors = torch.randn(10, ENCODING_DIM)
with torch.no_grad():
    random_images = model.decoder(latent_vectors)
assert random_images.shape == (10, 3, 24, 24), f"Expected shape (10, 3, 24, 24), but got {random_images.shape}"

# Save the 10 random images in one figure

fig = plt.figure(figsize=(10, 1))

for i in range(10):
    ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(np.transpose(random_images[i].detach().numpy(), (1, 2, 0)))

plt.savefig(f"{vis_root}/random_images_{args.model}.png")
