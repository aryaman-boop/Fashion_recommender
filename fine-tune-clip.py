import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import multiprocessing
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from typing import List
from comet_ml import Experiment
import os
import shutil
import glob
import time
import random
import math
import zipfile


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import data_utils  # initial import
from data_utils import base_path, targetpad_transform, FashionIQDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device


# Set your parameters here
dataset = "fashioniq"
api_key = None
workspace = None
experiment_name = "CLIP_finetuning_test"
num_epochs = 150
clip_model_name = "RN50x4"
encoder = 'both'
learning_rate = 2e-6
batch_size = 16
validation_frequency = 1
target_ratio = 1.25
transform = "targetpad"
save_training = True
save_best = True

# Now create the training_hyper_params dictionary
training_hyper_params = {
    "num_epochs": num_epochs,
    "clip_model_name": clip_model_name,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "validation_frequency": validation_frequency,
    "transform": transform,
    "target_ratio": target_ratio,
    "save_training": save_training,
    "encoder": encoder,
    "save_best": save_best
}

def clip_finetune_fiq(train_dress_types: List[str], val_dress_types: List[str],
                      num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, encoder: str, save_best: bool,
                      **kwargs):
    """
    Fine-tune CLIP on the FashionIQ dataset using as combining function the image-text element-wise sum
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning leanring rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the fine-tuned CLIP model
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best CLIP model wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_fiq_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

    if encoder == 'text':
        print('Only the CLIP text encoder will be fine-tuned')
        for param in clip_model.visual.parameters():
            param.requires_grad = False
    elif encoder == 'image':
        print('Only the CLIP image encoder will be fine-tuned')
        for param in clip_model.parameters():
            param.requires_grad = False
        for param in clip_model.visual.parameters():
            param.requires_grad = True
    elif encoder == 'both':
        print('Both CLIP encoders will be fine-tuned')
    else:
        raise ValueError("encoder parameter should be in ['text', 'image', both']")

    clip_model.eval().float()
    input_dim = clip_model.visual.input_resolution

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs
    if encoder == 'text':
        index_features_list = []
        index_names_list = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)
        if encoder == 'text':
            index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
            index_features_list.append(index_features_and_names[0])
            index_names_list.append(index_features_and_names[1])

    # Define the train datasets and the combining function
    print("Before dataset creation")
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    print("After dataset creation")
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=multiprocessing.cpu_count(), pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)
    combining_function = element_wise_sum

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, clip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                flattened_captions: list = np.array(captions).T.flatten().tolist()
                captions = generate_randomized_fiq_caption(flattened_captions)
                text_inputs = clip.tokenize(captions, context_length=77, truncate=True).to(device, non_blocking=True)

                # Extract the features, compute the logits and the loss
                with torch.cuda.amp.autocast():
                    reference_features = clip_model.encode_image(reference_images)
                    caption_features = clip_model.encode_text(text_inputs)
                    predicted_features = combining_function(reference_features, caption_features)
                    target_features = F.normalize(clip_model.encode_image(target_images))

                    logits = 100 * predicted_features @ target_features.T

                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)

                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            print('validation has not been implemented yet')

            if save_training:
                if save_best:
                    save_model('tuned_clip_best', epoch, clip_model, training_path)
                elif not save_best:
                    save_model(f'tuned_clip_{epoch}', epoch, clip_model, training_path)


# Configure Comet ML experiment
if api_key and workspace:
    print("Comet logging ENABLED")
    experiment = Experiment(
        api_key=api_key,
        project_name=f"{dataset} clip fine-tuning",
        workspace=workspace,
        disabled=False
    )
    if experiment_name:
        experiment.set_name(experiment_name)
else:
    print("Comet logging DISABLED, in order to enable it you need to provide an api key and a workspace")
    experiment = Experiment(
        api_key="",
        project_name="",
        workspace="",
        disabled=True
    )

# Try to log code to Comet (if enabled)
try:
    experiment.log_code(folder=str(base_path / 'src'))
except:
    print("Could not log code to Comet")

experiment.log_parameters(training_hyper_params)

clip_finetune_fiq(
    train_dress_types=['dress', 'shirt', 'toptee'],
    val_dress_types=['dress', 'shirt', 'toptee'],
    num_epochs=num_epochs,
    clip_model_name=clip_model_name,
    learning_rate=learning_rate,
    batch_size=batch_size,
    validation_frequency=validation_frequency,
    transform=transform,
    save_training=save_training,
    encoder=encoder,
    save_best=save_best,
    target_ratio=target_ratio
)