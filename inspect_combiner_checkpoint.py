import torch

ckpt_path = "checkpoint_epoch_10.pt"  # Update path if needed
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Keys in checkpoint:", list(ckpt.keys()))

state_dict = ckpt["model_state_dict"]
print("\nParameter shapes in model_state_dict:")
for k, v in state_dict.items():
    print(f"{k}: {tuple(v.shape)}") 