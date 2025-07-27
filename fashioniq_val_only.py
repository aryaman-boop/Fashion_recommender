import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import  FashionIQDataset, targetpad_transform
from test2 import DeepMLPCombiner
from utils import extract_index_features, collate_fn, element_wise_sum, device


def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, clip_model: CLIP, index_features: torch.tensor,
                            index_names: List[str], combining_function: callable) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_fiq_val_predictions(clip_model, relative_val_dataset,
                                                                    combining_function, index_names, index_features)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    
    # Check if we have exactly one label per target (accounting for skipped samples)
    labels_per_target = torch.sum(labels, dim=-1).int()
    expected_labels = torch.ones(len(target_names)).int()
    
    if not torch.equal(labels_per_target, expected_labels):
        print(f"Warning: Label mismatch detected. Expected {len(target_names)} targets with 1 label each, but got {labels_per_target.sum().item()} total labels.")
        print(f"Labels per target: {labels_per_target}")
        print(f"This may be due to missing reference images that were skipped during prediction.")
        
        # Filter out targets that don't have exactly one label
        valid_targets = labels_per_target == 1
        if valid_targets.sum() == 0:
            print("Error: No valid targets found. Cannot compute metrics.")
            return 0.0, 0.0
        
        labels = labels[valid_targets]
        target_names = [target_names[i] for i in range(len(target_names)) if valid_targets[i]]
        print(f"Proceeding with {valid_targets.sum().item()} valid targets out of {len(valid_targets)} total targets.")

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(clip_model: CLIP, relative_val_dataset: FashionIQDataset,
                                 combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, clip_model.module.visual.output_dim if hasattr(clip_model, 'module') else clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    skipped_samples = 0

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.module.encode_text(text_inputs) if hasattr(clip_model, 'module') else clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                # Fix: Add error handling for missing reference images
                try:
                    reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
                except KeyError as e:
                    print(f"Warning: Missing reference image {e} in index. Skipping this sample.")
                    skipped_samples += 1
                    continue
            else:
                # Fix: Add error handling for missing reference images
                try:
                    reference_image_features = torch.stack(itemgetter(*reference_names)(
                        name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                except KeyError as e:
                    print(f"Warning: Missing reference image {e} in index. Skipping this sample.")
                    skipped_samples += 1
                    continue
            
            # REVERT to original
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    if skipped_samples > 0:
        print(f"Summary: Skipped {skipped_samples} samples due to missing reference images.")

    return predicted_features, target_names


def fashioniq_val_retrieval(dress_type: str, combining_function: callable, clip_model: CLIP, preprocess: callable):
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                   combining_function)


def main():
    parser = ArgumentParser()
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--dress-type", default="dress", type=str, help="FashionIQ category: dress, shirt, or toptee")

    args = parser.parse_args()

    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.module.visual.input_resolution if hasattr(clip_model, 'module') else clip_model.visual.input_resolution
    feature_dim = clip_model.module.visual.output_dim if hasattr(clip_model, 'module') else clip_model.visual.output_dim

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
   
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = DeepMLPCombiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    recallat10, recallat50 = fashioniq_val_retrieval(args.dress_type, combining_function, clip_model, preprocess)
    print(f"\n{args.dress_type} recall@10 = {recallat10}")
    print(f"{args.dress_type} recall@50 = {recallat50}")

if __name__ == '__main__':
    main() 