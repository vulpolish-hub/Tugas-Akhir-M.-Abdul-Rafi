import json

path = "kaggle_sen12_gan_resnet/sen12ms-cr-ts-cloud-removal-gan.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ResNet-18 structure with Spectral Normalization
disc_code = [
    "import torch.nn.utils.spectral_norm as spectral_norm\n\n",
    "class ResidualBlock(nn.Module):\n",
    "    def __init__(self, in_ch, out_ch, stride=1):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            spectral_norm(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)),\n",
    "            nn.BatchNorm2d(out_ch),\n",
    "            nn.ReLU(inplace=True),\n",
    "            spectral_norm(nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)),\n",
    "            nn.BatchNorm2d(out_ch),\n",
    "        )\n",
    "        self.skip = nn.Sequential(\n",
    "            spectral_norm(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)),\n",
    "            nn.BatchNorm2d(out_ch)\n",
    "        ) if stride != 1 or in_ch != out_ch else nn.Identity()\n",
    "        self.act = nn.ReLU(inplace=True)\n\n",
    "    def forward(self, x):\n",
    "        return self.act(self.net(x) + self.skip(x))\n\n",
    "class ResNetDiscriminator(nn.Module):\n",
    "    def __init__(self, in_channels=13, base=64):\n",
    "        super().__init__()\n",
    "        self.init_conv = nn.Sequential(\n",
    "            spectral_norm(nn.Conv2d(in_channels, base, 7, stride=2, padding=3, bias=False)),\n",
    "            nn.BatchNorm2d(base),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.MaxPool2d(3, stride=2, padding=1)\n",
    "        )\n",
    "        self.layer1 = ResidualBlock(base, base)\n",
    "        self.layer2 = ResidualBlock(base, base*2, stride=2)\n",
    "        self.layer3 = ResidualBlock(base*2, base*4, stride=2)\n",
    "        self.gap = nn.AdaptiveAvgPool2d((1, 1))\n",
    "        self.fc = nn.Linear(base*4, 1)\n\n",
    "    def forward(self, x):\n",
    "        x = self.init_conv(x)\n",
    "        x = self.layer1(x)\n",
    "        x = self.layer2(x)\n",
    "        x = self.layer3(x)\n",
    "        x = self.gap(x)\n",
    "        x = torch.flatten(x, 1)\n",
    "        return self.fc(x)\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": disc_code
}

# Find cell index containing MultiTemporalResUNet
target_idx = -1
for idx, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    if "class MultiTemporalResUNet" in src:
        target_idx = idx
        break

if target_idx != -1:
    nb['cells'].insert(target_idx + 1, new_cell)
    print(f"Discriminator inserted at cell index {target_idx + 1}")
else:
    # fallback: append to end
    nb['cells'].append(new_cell)
    print("MultiTemporalResUNet cell not found, appended Discriminator to the end")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook modification complete.")
