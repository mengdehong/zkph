"""
This module provides functionality for loading a pre-trained DINOv2 model and generating perceptual hashes for images.

Classes:
    Hash:
        A class to handle the perceptual hash tensor and provide various conversion methods.
        Methods:
            __init__(tensor: torch.Tensor):
                Initializes the Hash object with a given tensor.
            to_hex() -> str:
                Converts the hash tensor to a hexadecimal string.
            to_string() -> str:
                Converts the hash tensor to a binary string.
            to_pytorch() -> torch.Tensor:
                Returns the hash tensor as a PyTorch tensor.
            to_numpy() -> np.ndarray:
                Returns the hash tensor as a NumPy array.

Functions:
    load_model(path: str) -> None:
        Loads the DINOv2 model state from the specified path.
    hash(image_arrays: Union[np.ndarray, List[Image.Image]]) -> torch.Tensor:
        Generates perceptual hashes for the given images using the DINOv2 model.
        Parameters:
            image_arrays (Union[np.ndarray, List[Image.Image], torch.Tensor]): Input images as a numpy array or a list of PIL Images or a torch.Tensor.
            differentiable (bool): If True, enables gradient computation. Default is False.
            mydinov2 (torch.nn.Module): The DINOv2 model to use for generating hashes. Default is the globally loaded model.
        Returns:
            torch.Tensor: The generated perceptual hashes.
"""

import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from typing import Union, List
import sys

class DINOHash:
    def __init__(self, pca_dims=96, model="vits14_reg", prod_mode=True):
        self.pca_dims = pca_dims
        self.model = model
        self.prod_mode = prod_mode
        # Prefer timm implementation to avoid Python 3.10-only syntax in upstream hub code
        try:
            import timm
            # Create model without pretrained weights first. We'll try to load a local safetensors
            # checkpoint from the project's checkpoint/ directory to avoid network downloads.
            # If local checkpoint not present or fails to load, fall back to timm pretrained behavior.
            try:
                self.dinov2 = timm.create_model('vit_small_patch14_dinov2', pretrained=False, num_classes=0, img_size=224)
                # safety: ensure patch embed expects 224x224
                try:
                    if hasattr(self.dinov2, 'patch_embed') and hasattr(self.dinov2.patch_embed, 'img_size'):
                        self.dinov2.patch_embed.img_size = (224, 224)
                except Exception:
                    pass

                # Look for local safetensors in project checkpoint/ (one level up from this file)
                base_file_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(base_file_dir, '..'))
                local_safetensors = os.path.join(project_root, 'checkpoint', f'dinov2_{self.model}.safetensors')
                if os.path.exists(local_safetensors):
                    try:
                        from safetensors.torch import load_file as _load_safetensors
                        st = _load_safetensors(local_safetensors)
                        # convert values to torch.Tensor if needed
                        st_t = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in st.items()}
                        self.dinov2.load_state_dict(st_t, strict=False)
                        print(f"Loaded local safetensors for dinov2 from: {local_safetensors}")
                    except Exception as e:
                        print(f"safetensors load failed ({local_safetensors}): {e}. Trying torch.load fallback...")
                        try:
                            sd = torch.load(local_safetensors, map_location='cpu')
                            # if wrapped in dict like {'state_dict': ...}
                            if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
                                sd = sd['state_dict']
                            self.dinov2.load_state_dict(sd, strict=False)
                            print(f"Loaded local torch checkpoint for dinov2 from: {local_safetensors}")
                        except Exception as e2:
                            print(f"Local checkpoint load failed: {e2}. Falling back to timm pretrained download.")
                            # fallback to timm pretrained (may download)
                            self.dinov2 = timm.create_model('vit_small_patch14_dinov2', pretrained=True, num_classes=0, img_size=224)
                else:
                    # no local checkpoint: fall back to timm pretrained (may download)
                    self.dinov2 = timm.create_model('vit_small_patch14_dinov2', pretrained=True, num_classes=0, img_size=224)
            except Exception:
                # if anything goes wrong with timm flow, raise to outer except to try hub
                raise
        except Exception:
            # Fallback to torch.hub if timm is not available
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_{self.model}')
        self.dinov2 = self.dinov2.cuda()
        for param in self.dinov2.parameters():
            param.requires_grad = False
        self.dinov2.eval()

        # Build instance-level preprocess to match model's expected resolution
        try:
            cfg = getattr(self.dinov2, 'pretrained_cfg', {}) or {}
            input_size = cfg.get('input_size', (3, 224, 224))
            target_h, target_w = int(input_size[1]), int(input_size[2])
        except Exception:
            target_h = target_w = 224
        self.preprocess = transforms.Compose([
            transforms.Resize((target_h, target_w)),
            transforms.ToTensor(),
        ])

        base_dir = os.path.dirname(os.path.abspath(__file__))
        means_path = os.path.join(base_dir, f'dinov2_{self.model}_means.npy')
        pca_path = os.path.join(base_dir, f'dinov2_{self.model}_PCA.npy')
        
        # Try to load pre-computed PCA, but allow graceful fallback
        try:
            if os.path.exists(means_path) and os.path.exists(pca_path):
                means = np.load(means_path)
                self.means_torch = torch.from_numpy(means).cuda().float()
                
                components = np.load(pca_path).T
                self.components_torch = torch.from_numpy(components).cuda().float()
                print(f"Loaded pre-computed PCA for {self.model}")
            else:
                # Initialize with placeholder values - will be overridden by eval scripts
                print(f"Pre-computed PCA not found for {self.model}, using placeholder values")
                self.means_torch = torch.zeros(384).cuda().float()  # DINOv2 vits14 feature dim
                self.components_torch = torch.eye(384, self.pca_dims).cuda().float()
        except Exception as e:
            print(f"Warning: Failed to load PCA files: {e}")
            # Initialize with placeholder values
            self.means_torch = torch.zeros(384).cuda().float()
            self.components_torch = torch.eye(384, self.pca_dims).cuda().float()
    
    def load_model(self, path):
        self.dinov2.load_state_dict(torch.load(path, weights_only=True))
    
    def hash(
        self,
        image_arrays: Union[np.ndarray, List[Image.Image], torch.Tensor],
        differentiable: bool = False,
        c: int = 1,
        logits: bool = False,
        l2_normalize: bool = False,
        ) -> torch.Tensor:

        wrapper = torch.no_grad if not differentiable else torch.enable_grad

        if isinstance(image_arrays, np.ndarray):
            image_arrays = torch.from_numpy(image_arrays)
        if isinstance(image_arrays[0], Image.Image):
            image_arrays = torch.stack([self.preprocess(im) for im in image_arrays])
        if isinstance(image_arrays[0], str):
            image_arrays = torch.stack([self.preprocess(Image.open(im)) for im in image_arrays])

        with wrapper():
            image_arrays = normalize(image_arrays.cuda())
            
            outs = self.dinov2(image_arrays) - self.means_torch
            
            outs = outs @ self.components_torch

            outs = outs[:, :self.pca_dims]

            if l2_normalize:
                outs = torch.nn.functional.normalize(outs, dim=1)
            outs *= c

            if not logits:
                if differentiable:
                    outs = torch.sigmoid(outs)
                else:
                    outs = outs >= 0

        del image_arrays

        if self.prod_mode:
            return [Hash(out) for out in outs]
        return outs

class Hash:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.cpu()
        self.string = ''.join(str(int(x)) for x in self.tensor)
        self.hex = hex(int(self.string, 2))
        self.array = self.tensor.numpy()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Note: per-instance preprocess is built in DINOHash.__init__ based on model cfg.

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dinohash.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path)
    dinohash = DINOHash(pca_dims=96, model="vits14_reg", prod_mode=True).hash
    hash_tensor = dinohash([image])[0].hex
    print("Perceptual hash:", hash_tensor)

