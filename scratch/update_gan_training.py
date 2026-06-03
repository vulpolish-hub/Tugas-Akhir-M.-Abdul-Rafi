import json

path = "kaggle_sen12_gan_resnet/sen12ms-cr-ts-cloud-removal-gan.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the training loop cell
train_cell_idx = -1
for idx, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    if "def train_one_epoch" in src:
        train_cell_idx = idx
        break

if train_cell_idx != -1:
    print(f"Found training cell at index {train_cell_idx}. Modifying it...")
    
    new_source = [
        "def mae(pred, target):\n",
        "    return F.l1_loss(pred, target)\n\n",
        "def masked_mae(pred, target, mask):\n",
        "    diff = torch.abs(pred - target) * mask\n",
        "    return diff.sum() / (mask.sum() * pred.shape[1] + 1e-8)\n\n",
        "def weighted_masked_mae(pred, target, mask, weight):\n",
        "    diff = torch.abs(pred - target) * mask * weight\n",
        "    return diff.sum() / ((mask * weight).sum() * pred.shape[1] + 1e-8)\n\n",
        "def bucket_mae(pred, target, bucket_mask):\n",
        "    diff = torch.abs(pred - target) * bucket_mask\n",
        "    val = diff.sum() / (bucket_mask.sum() * pred.shape[1] + 1e-8)\n",
        "    return torch.where(bucket_mask.sum() > 0, val, torch.tensor(float('nan'), device=pred.device))\n\n",
        "def gradient_mae(pred, target, mask):\n",
        "    # spatial gradient loss\n",
        "    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])\n",
        "    dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])\n",
        "    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])\n",
        "    dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])\n",
        "    \n",
        "    mask_y = mask[:, :, 1:, :]\n",
        "    mask_x = mask[:, :, :, 1:]\n",
        "    \n",
        "    loss_y = torch.abs(dy_pred - dy_target) * mask_y\n",
        "    loss_x = torch.abs(dx_pred - dx_target) * mask_x\n",
        "    \n",
        "    total_loss = loss_y.sum() / (mask_y.sum() * pred.shape[1] + 1e-8) + loss_x.sum() / (mask_x.sum() * pred.shape[1] + 1e-8)\n",
        "    return total_loss\n\n",
        "def rmse(pred, target):\n",
        "    return torch.sqrt(F.mse_loss(pred, target))\n\n",
        "def psnr(pred, target, max_val=1.0):\n",
        "    mse = F.mse_loss(pred, target)\n",
        "    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse + 1e-8)\n\n",
        "def apply_copy_outside_mask(raw_pred, t0, mask):\n",
        "    return raw_pred * mask + t0 * (1.0 - mask)\n\n",
        "def heavy_cloud_weight(mask):\n",
        "    return 1.0 + CFG.HEAVY_LOSS_BOOST * mask + CFG.HEAVY_BINARY_BOOST * (mask >= CFG.HEAVY_PIXEL_THRESHOLD).float()\n\n",
        "def cloud_bucket_masks(mask):\n",
        "    light = (mask > 0.0) & (mask < 0.33)\n",
        "    medium = (mask >= 0.33) & (mask < 0.66)\n",
        "    heavy = (mask >= 0.66)\n",
        "    return light.float(), medium.float(), heavy.float()\n\n",
        "def scalar_or_none(x):\n",
        "    return float(x.detach().cpu()) if not torch.isnan(x) else None\n\n",
        "def evaluate(loader, label=\"val\"):\n",
        "    model.eval()\n",
        "    netD.eval()\n",
        "    metrics = {\n",
        "        \"model_mae\": [], \"model_cloud_mae\": [], \"model_light_cloud_mae\": [],\n",
        "        \"model_medium_cloud_mae\": [], \"model_heavy_cloud_mae\": [],\n",
        "        \"raw_cloud_mae\": [], \"t0_mae\": [], \"t0_cloud_mae\": [],\n",
        "        \"t0_light_cloud_mae\": [], \"t0_medium_cloud_mae\": [], \"t0_heavy_cloud_mae\": [],\n",
        "        \"rmse\": [], \"psnr\": [], \"cloud_fraction\": [], \"heavy_fraction\": []\n",
        "    }\n",
        "    with torch.no_grad():\n",
        "        for batch in loader:\n",
        "            x = batch[\"x\"].to(DEVICE)\n",
        "            y = batch[\"y\"].to(DEVICE)\n",
        "            mask = batch[\"cloud_mask\"].to(DEVICE)\n",
        "            t0 = batch[\"t0\"].to(DEVICE)\n",
        "            \n",
        "            raw = model(x)\n",
        "            pred = apply_copy_outside_mask(raw, t0, mask)\n",
        "            \n",
        "            light_m, medium_m, heavy_m = cloud_bucket_masks(mask)\n",
        "            \n",
        "            metrics[\"model_mae\"].append(float(F.l1_loss(pred, y).cpu()))\n",
        "            metrics[\"model_cloud_mae\"].append(float(masked_mae(pred, y, mask).cpu()))\n",
        "            metrics[\"model_light_cloud_mae\"].append(scalar_or_none(bucket_mae(pred, y, light_m)))\n",
        "            metrics[\"model_medium_cloud_mae\"].append(scalar_or_none(bucket_mae(pred, y, medium_m)))\n",
        "            metrics[\"model_heavy_cloud_mae\"].append(scalar_or_none(bucket_mae(pred, y, heavy_m)))\n",
        "            metrics[\"raw_cloud_mae\"].append(float(masked_mae(raw, y, mask).cpu()))\n",
        "            \n",
        "            metrics[\"t0_mae\"].append(float(F.l1_loss(t0, y).cpu()))\n",
        "            metrics[\"t0_cloud_mae\"].append(float(masked_mae(t0, y, mask).cpu()))\n",
        "            metrics[\"t0_light_cloud_mae\"].append(scalar_or_none(bucket_mae(t0, y, light_m)))\n",
        "            metrics[\"t0_medium_cloud_mae\"].append(scalar_or_none(bucket_mae(t0, y, medium_m)))\n",
        "            metrics[\"t0_heavy_cloud_mae\"].append(scalar_or_none(bucket_mae(t0, y, heavy_m)))\n",
        "            \n",
        "            metrics[\"rmse\"].append(float(rmse(pred, y).cpu()))\n",
        "            metrics[\"psnr\"].append(float(psnr(pred, y).cpu()))\n",
        "            metrics[\"cloud_fraction\"].append(float(mask.mean().cpu()))\n",
        "            metrics[\"heavy_fraction\"].append(float(heavy_m.mean().cpu()))\n",
        "            \n",
        "    res = {}\n",
        "    for k, vals in metrics.items():\n",
        "        clean = [v for v in vals if v is not None]\n",
        "        res[k] = float(np.mean(clean)) if clean else 0.0\n",
        "    return res\n\n",
        "def train_one_epoch(loader, epoch, total_epochs):\n",
        "    model.train()\n",
        "    netD.train()\n",
        "    losses = []\n",
        "    d_losses = []\n",
        "    heavy_losses = []\n",
        "    iterator = tqdm(loader, desc=f\"epoch {epoch:03d}/{total_epochs:03d}\", leave=True)\n",
        "    for batch in iterator:\n",
        "        x = batch[\"x\"].to(DEVICE)\n",
        "        y = batch[\"y\"].to(DEVICE)\n",
        "        mask = batch[\"cloud_mask\"].to(DEVICE)\n",
        "        t0 = batch[\"t0\"].to(DEVICE)\n",
        "        \n",
        "        # 1. Update Discriminator\n",
        "        optimizerD.zero_grad()\n",
        "        # Real\n",
        "        out_real = netD(y)\n",
        "        loss_D_real = criterion_GAN(out_real, torch.ones_like(out_real))\n",
        "        # Fake\n",
        "        raw = model(x)\n",
        "        pred = apply_copy_outside_mask(raw, t0, mask)\n",
        "        out_fake = netD(pred.detach())\n",
        "        loss_D_fake = criterion_GAN(out_fake, torch.zeros_like(out_fake))\n",
        "        \n",
        "        loss_D = (loss_D_real + loss_D_fake) * 0.5\n",
        "        loss_D.backward()\n",
        "        optimizerD.step()\n",
        "        d_losses.append(float(loss_D.detach().cpu()))\n",
        "        \n",
        "        # 2. Update Generator\n",
        "        optimizer.zero_grad(set_to_none=True)\n",
        "        out_fake_G = netD(pred)\n",
        "        loss_G_adv = criterion_GAN(out_fake_G, torch.ones_like(out_fake_G))\n",
        "        \n",
        "        weight = heavy_cloud_weight(mask)\n",
        "        whole = F.l1_loss(pred, y)\n",
        "        cloud_plain = masked_mae(pred, y, mask)\n",
        "        cloud_weighted = weighted_masked_mae(pred, y, mask, weight)\n",
        "        raw_cloud_weighted = weighted_masked_mae(raw, y, mask, weight)\n",
        "        copy_penalty = masked_mae(pred, t0, 1.0 - mask)\n",
        "        grad = gradient_mae(pred, y, mask)\n",
        "        _, _, heavy_m = cloud_bucket_masks(mask)\n",
        "        heavy_only = bucket_mae(pred, y, heavy_m)\n",
        "        heavy_term = torch.where(torch.isnan(heavy_only), cloud_plain, heavy_only)\n",
        "        \n",
        "        loss = (\n",
        "            0.15 * whole\n",
        "            + 1.15 * cloud_weighted\n",
        "            + 0.20 * raw_cloud_weighted\n",
        "            + 0.25 * copy_penalty\n",
        "            + 0.10 * grad\n",
        "            + 0.15 * heavy_term\n",
        "            + 0.05 * loss_G_adv\n",
        "        )\n",
        "        loss.backward()\n",
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
        "        optimizer.step()\n",
        "        losses.append(float(loss.detach().cpu()))\n",
        "        heavy_losses.append(0.0 if bool(torch.isnan(heavy_only).detach().cpu()) else float(heavy_only.detach().cpu()))\n",
        "        \n",
        "        iterator.set_postfix(loss_G=np.mean(losses), loss_D=np.mean(d_losses))\n",
        "        \n",
        "    return float(np.mean(losses)), float(np.mean(d_losses)), float(np.mean(heavy_losses))\n\n",
        "def save_training_plot(history_df):\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "    axes[0].plot(history_df[\"epoch\"], history_df[\"train_loss\"], marker=\"o\", label=\"train G loss\")\n",
        "    axes[0].plot(history_df[\"epoch\"], history_df[\"train_cloud_loss\"], marker=\"o\", label=\"train D loss\")\n",
        "    axes[0].plot(history_df[\"epoch\"], history_df[\"val_model_mae\"], marker=\"x\", label=\"val model MAE\")\n",
        "    axes[0].set_title(\"Training Loss & Val MAE\")\n",
        "    axes[0].legend()\n",
        "    axes[0].grid(True)\n",
        "    \n",
        "    axes[1].plot(history_df[\"epoch\"], history_df[\"val_model_cloud_mae\"], marker=\"o\", label=\"val cloud MAE\")\n",
        "    axes[1].plot(history_df[\"epoch\"], history_df[\"val_model_heavy_cloud_mae\"], marker=\"x\", label=\"val heavy cloud MAE\")\n",
        "    axes[1].set_title(\"Validation Cloud MAE\")\n",
        "    axes[1].legend()\n",
        "    axes[1].grid(True)\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(CFG.OUT_DIR / \"training_curves.png\", dpi=150)\n",
        "    plt.close()\n\n",
        "print(\"strategy: hard/soft copy outside mask; spectral-normalized ResNet-18 GAN; heavy-cloud oversampling\")\n",
        "\n",
        "if CFG.RUN_TRAINING:\n",
        "    history = []\n",
        "    best_cloud_mae = float(\"inf\")\n",
        "    \n",
        "    for epoch in range(1, CFG.EPOCHS + 1):\n",
        "        train_loss, train_d_loss, train_heavy_loss = train_one_epoch(train_loader, epoch, CFG.EPOCHS)\n",
        "        val_metrics = evaluate(val_loader, label=\"val\")\n",
        "        \n",
        "        log_row = {\n",
        "            \"epoch\": epoch,\n",
        "            \"train_loss\": train_loss,\n",
        "            \"train_cloud_loss\": train_d_loss,\n",
        "            \"train_heavy_loss\": train_heavy_loss,\n",
        "            \"lr\": optimizer.param_groups[0][\"lr\"],\n",
        "        }\n",
        "        log_row.update(val_metrics)\n",
        "        history.append(log_row)\n",
        "        \n",
        "        # Print metrics\n",
        "        print(f\"Epoch {epoch:02d} | Train G: {train_loss:.5f} | Train D: {train_d_loss:.5f} | Val Cloud MAE: {val_metrics['val_model_cloud_mae']:.5f} | PSNR: {val_metrics['psnr']:.2f} dB\")\n",
        "        \n",
        "        # Checkpoint\n",
        "        if val_metrics[\"val_model_cloud_mae\"] < best_cloud_mae:\n",
        "            best_cloud_mae = val_metrics[\"val_model_cloud_mae\"]\n",
        "            torch.save(model.state_dict(), CFG.OUT_DIR / \"best_multitemporal_resunet_hardmask.pth\")\n",
        "            torch.save(netD.state_dict(), CFG.OUT_DIR / \"best_resnet_discriminator.pth\")\n",
        "            print(f\"  ==> Saved best model weights with cloud MAE {best_cloud_mae:.6f}\")\n",
        "            \n",
        "        torch.save(model.state_dict(), CFG.OUT_DIR / \"last_multitemporal_resunet_hardmask.pth\")\n",
        "        \n",
        "        # Save history periodically\n",
        "        history_df = pd.DataFrame(history)\n",
        "        history_df.to_csv(CFG.OUT_DIR / \"training_history.csv\", index=False)\n",
        "        save_training_plot(history_df)\n"
    ]
    
    nb['cells'][train_cell_idx]['source'] = new_source
    print("Training cell modified successfully.")
else:
    print("Error: train_one_epoch cell not found!")

# Now find where "model = MultiTemporalResUNet" is defined to append the netD definition
init_cell_idx = -1
for idx, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    if "model = MultiTemporalResUNet" in src:
        init_cell_idx = idx
        break

if init_cell_idx != -1:
    print(f"Found initialization cell at index {init_cell_idx}. Modifying it...")
    
    src_lines = nb['cells'][init_cell_idx]['source']
    if isinstance(src_lines, str):
        src_lines = src_lines.split("\n")
    
    # check if netD is already there
    has_netD = False
    for line in src_lines:
        if "netD =" in line:
            has_netD = True
            break
            
    if not has_netD:
        src_lines.extend([
            "\n# GAN Initialization\n",
            "netD = ResNetDiscriminator(in_channels=13, base=CFG.BASE_CHANNELS).to(DEVICE)\n",
            "optimizerD = torch.optim.Adam(netD.parameters(), lr=CFG.LR, betas=(0.5, 0.999))\n",
            "criterion_GAN = nn.BCEWithLogitsLoss()\n",
            "print('Discriminator and GAN losses initialized.')\n"
        ])
        nb['cells'][init_cell_idx]['source'] = src_lines
        print("Initialization cell modified successfully.")
    else:
        print("netD already initialized in this cell.")
else:
    print("Error: Model initialization cell not found!")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook modification complete.")
