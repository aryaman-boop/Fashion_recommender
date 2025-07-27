import json
from pathlib import Path
from datetime import datetime
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from statistics import mean
import torch.nn.functional as F
import glob

from data_utils import base_path, targetpad_transform, FashionIQDataset
from test2 import DeepMLPCombiner
from attention_fusion_combiner import AttentionFusionCombiner
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, extract_index_features, generate_randomized_fiq_caption, device
from fashioniq_val_only import fashioniq_val_retrieval

# --- Settings ---
clip_model_name = "RN50x4"
clip_model_path = None  # Set path if using a fine-tuned CLIP
projection_dim = 512
hidden_dim = 512
num_epochs = 50
combiner_lr = 1e-4
batch_size = 64
clip_bs = 32
validation_frequency = 1
target_ratio = 1.25
transform = "targetpad"
save_training = True
save_best = True

train_dress_types = ['dress', 'shirt', 'toptee']
val_dress_types = ['dress', 'shirt', 'toptee']

training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
output_dir = Path("vector_db_gnn_outputs")
training_path = Path(base_path / f"models/combiner_trained_on_fiq_{clip_model_name}_{training_start}")
training_path.mkdir(exist_ok=False, parents=True)

# Save hyperparameters
training_hyper_params = {
    "projection_dim": projection_dim,
    "hidden_dim": hidden_dim,
    "num_epochs": num_epochs,
    "clip_model_name": clip_model_name,
    "clip_model_path": clip_model_path,
    "combiner_lr": combiner_lr,
    "batch_size": batch_size,
    "clip_bs": clip_bs,
    "validation_frequency": validation_frequency,
    "transform": transform,
    "target_ratio": target_ratio,
    "save_training": save_training,
    "save_best": save_best,
    "train_dress_types": train_dress_types,
    "val_dress_types": val_dress_types,
}
with open(training_path / "training_hyperparameters.json", 'w+') as file:
    json.dump(training_hyper_params, file, sort_keys=True, indent=4)

# --- Load CLIP model ---
import clip
clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
clip_model.eval()
clip_model_path = base_path / f"models/clip_finetuned_on_fiq_RN50x4/saved_models/tuned_clip_best.pt"        
saved_state_dict = torch.load(clip_model_path, map_location=device)
clip_model.load_state_dict(saved_state_dict["CLIP"])  # Load fine-tuned weights
input_dim = clip_model.visual.input_resolution
feature_dim = clip_model.visual.output_dim

if transform == "targetpad":
    preprocess = targetpad_transform(target_ratio, input_dim)
else:
    preprocess = clip_preprocess



clip_model = clip_model.float()
idx_to_dress_mapping = ['dress', 'shirt', 'toptee']
# --- Prepare datasets ---
relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True, shuffle=True)

# After creating your DataLoader, before the training loop:
sample = next(iter(relative_train_loader))
ref_imgs, tgt_imgs, captions = sample
print("Sample reference image tensor shape:", ref_imgs.shape)
print("Sample target image tensor shape:", tgt_imgs.shape)
print("Sample captions:", captions[:2])

relative_val_dataset = FashionIQDataset('val', val_dress_types, 'relative', preprocess)
classic_val_dataset = FashionIQDataset('val', val_dress_types, 'classic', preprocess)
val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)

# --- Initialize Combiner ---
# Choose which combiner to use: 'mlp' or 'attention'
combiner_type = 'attention'  # options: 'mlp', 'attention'
if combiner_type == 'mlp':
    combiner = DeepMLPCombiner(feature_dim, projection_dim, hidden_dim).to(device)
else:
    combiner = AttentionFusionCombiner(feature_dim, projection_dim, hidden_dim).to(device)
optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)
scaler = torch.cuda.amp.GradScaler()
best_avg_recall = 0

training_log_frame = pd.DataFrame()
validation_log_frame = pd.DataFrame()

# --- Checkpoint resume settings ---
resume_from_checkpoint = True  # Set to True to resume, False to start fresh
# Automatically find the latest checkpoint if available
checkpoint_files = sorted(glob.glob(str(training_path / 'checkpoint_epoch_*.pt')),
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))
checkpoint_path = checkpoint_files[-1] if (resume_from_checkpoint and checkpoint_files) else None
start_epoch = 0

if resume_from_checkpoint and checkpoint_path:
    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    combiner.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    best_avg_recall = checkpoint.get('best_avg_recall', 0)
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0

# --- Batch-hard triplet loss function ---
def batch_hard_triplet_loss(anchor, positive, margin=0.2):
    sim_matrix = anchor @ positive.T  # (B, B)
    pos = sim_matrix.diag()
    mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
    sim_matrix_neg = sim_matrix.masked_fill(mask, float('-inf'))
    hardest_neg, _ = sim_matrix_neg.max(dim=1)
    loss = F.relu(margin + hardest_neg - pos).mean()
    return loss

# --- Training Loop ---
print('Training loop started')
for epoch in range(start_epoch, num_epochs):
    combiner.train()
    train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
    train_bar = tqdm(relative_train_loader, ncols=150)
    for idx, (reference_images, target_images, captions) in enumerate(train_bar):
        images_in_batch = reference_images.size(0)
        optimizer.zero_grad()
        reference_images = reference_images.to(device)
        target_images = target_images.to(device)
        flattened_captions = np.array(captions).T.flatten().tolist()
        input_captions = generate_randomized_fiq_caption(flattened_captions)
        text_inputs = clip.tokenize(input_captions, truncate=True).to(device)
        with torch.no_grad():
            reference_image_features = clip_model.encode_image(reference_images).float()
            target_image_features = clip_model.encode_image(target_images).float()
            text_features = clip_model.encode_text(text_inputs).float()
            ref_tgt_sim = F.cosine_similarity(reference_image_features, target_image_features).mean()
            print(f"[Diag] Avg. Ref-Tgt Similarity: {ref_tgt_sim.item():.4f}")
            print("reference_image_features.shape:", reference_image_features.shape)
            cos_txt_tgt = F.cosine_similarity(text_features, target_image_features, dim=-1)
            print(f"[Diag] cos(text, target) mean: {cos_txt_tgt.mean().item():.3f}, min: {cos_txt_tgt.min().item():.3f}, max: {cos_txt_tgt.max().item():.3f}")
        if epoch == 0 and idx < 2:
            print("Batch index:", idx)
            print("Reference image features[0]:", reference_image_features[0][:5].cpu().numpy())
            print("Target image features[0]:", target_image_features[0][:5].cpu().numpy())
            print("Text features[0]:", text_features[0][:5].cpu().numpy())
            print("Are reference and target images the same?", torch.allclose(reference_image_features[0], target_image_features[0]))
            print("Sample caption:", input_captions[0])

        with torch.cuda.amp.autocast():
            predicted_features = combiner.combine_features(reference_image_features, text_features)
            target_features = F.normalize(target_image_features, dim=-1)
            loss = batch_hard_triplet_loss(predicted_features, target_features, margin=0.2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        update_train_running_results(train_running_results, loss, images_in_batch)
        set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)
    train_epoch_loss = float(train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
    training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
    training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

    # --- Validation ---
    if epoch % validation_frequency == 0:
        clip_model = clip_model.float()  # In validation we use fp32 CLIP model
        combiner.eval()
        recalls_at10 = []
        recalls_at50 = []

        # Compute and log validation metrics for each validation dataset (which corresponds to a different FashionIQ category)
        for dress_type in idx_to_dress_mapping:
            recall_at10, recall_at50 = fashioniq_val_retrieval(
                dress_type=dress_type,
                combining_function=combiner.combine_features,
                clip_model=clip_model,
                preprocess=clip_preprocess
            )
            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)

        results_dict = {}
        for i, dress_type in enumerate(idx_to_dress_mapping):
            results_dict[f'{dress_type}_recall_at10'] = recalls_at10[i]
            results_dict[f'{dress_type}_recall_at50'] = recalls_at50[i]
        if recalls_at10 and recalls_at50:
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
            })
        else:
            print("Warning: No recall values computed. Check idx_to_dress_mapping and validation loop.")
            results_dict.update({
                f'average_recall_at10': None,
                f'average_recall_at50': None,
                f'average_recall': None
            })

        print(json.dumps(results_dict, indent=4))
        # Remove or comment out experiment.log_metrics if not using experiment tracking
        # experiment.log_metrics(results_dict, epoch=epoch)

        # Validation CSV logging
        log_dict = {'epoch': epoch}
        log_dict.update(results_dict)
        validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
        validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

        # Save model
        if save_training:
            if save_best and results_dict['average_recall'] > best_avg_recall:
                best_avg_recall = results_dict['average_recall']
                save_model('combiner', epoch, combiner, training_path)
            elif not save_best:
                save_model(f'combiner_{epoch}', epoch, combiner, training_path)

    print("idx_to_dress_mapping:", idx_to_dress_mapping)

    # --- Save checkpoint after each epoch ---
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': combiner.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_avg_recall': best_avg_recall,
    }
    torch.save(checkpoint, training_path / f'checkpoint_epoch_{epoch}.pt')

print("Training complete.")


