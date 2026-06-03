import os
import shutil

os.makedirs("kaggle_sen12_gan_resnet", exist_ok=True)
shutil.copy("kaggle_sen12_eda_unet/sen12ms-cr-ts-cloud-removal-eda-baseline-unet.ipynb", "kaggle_sen12_gan_resnet/sen12ms-cr-ts-cloud-removal-gan.ipynb")
print("Notebook copied successfully.")
