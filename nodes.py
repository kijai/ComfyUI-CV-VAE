import os
import torch

import folder_paths
import comfy.model_management as mm

from .models.modeling_vae import CVVAEModel

class CV_VAE_Load:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            },
        }

    RETURN_TYPES = ("CVVAE",)
    RETURN_NAMES = ("cvvae",)
    FUNCTION = "load"
    CATEGORY = "CV-VAE"

    def load(self):
        mm.soft_empty_cache()
        device = mm.vae_device()
        vae_dtype=mm.vae_dtype()
        base_path = os.path.join(folder_paths.models_dir, "vae")
        vae_path = os.path.join(base_path, "CV-VAE")
        if not os.path.exists(vae_path):
                print(f"Downloading model to: {vae_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="AILab-CVC/CV-VAE", 
                                  ignore_patterns=["*.ckpt"],
                                  local_dir=vae_path, 
                                  local_dir_use_symlinks=False)
        vae3d = CVVAEModel.from_pretrained(vae_path,subfolder="vae3d",torch_dtype=vae_dtype)
        vae3d.requires_grad_(False)
        vae3d.to(device)
        return (vae3d,)
    

class CV_VAE_Encode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "cvvae": ("CVVAE",),
            "images": ("IMAGE",),

            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "CV-VAE"

    def encode(self, cvvae, images):
        mm.soft_empty_cache()
        device = cvvae.device
        B, H, W, C = images.shape
        images = (images - 0.5) * 2.0

        images = images.permute(3, 0, 1, 2).to(cvvae.dtype).to(device)
        images = images.unsqueeze(0)
        print("input image shape: ", images.shape)

        frame_end = 1 + (B - 1) // 4 * 4
        print("frame end: ", frame_end)
        video= images[:,:,:frame_end,:,:]
        print("video shape: ", video.shape)
        latent = cvvae.encode(video).latent_dist.sample()
        print("latent shape: ", latent.shape)
       
        return ({"samples":latent},)

class CV_VAE_Decode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "cvvae": ("CVVAE",),
            "samples": ("LATENT",),

            },
            "optional": {
                "normalize": ("BOOLEAN", True),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "encode"
    CATEGORY = "CV-VAE"

    def encode(self, cvvae, samples, normalize=True):
        mm.soft_empty_cache()
        device = cvvae.device
     
        samples = samples["samples"]
        samples = samples.to(cvvae.dtype).to(device)
        print("samples shape: ",samples.shape)
        if (len(samples.shape) == 4):
            samples = samples.permute(1, 0, 2, 3)
            samples = samples.unsqueeze(0)
        if normalize:
            samples = (samples - 0.5) * 2.0
 
        images = cvvae.decode(samples).sample
        print("decoded image shape: ",images.shape)
 
        images = (torch.clamp(images,-1.0, 1.0) + 1.0) / 2.0
        images = images.squeeze(0)
        print(images.shape)
        images = images.permute(1, 2, 3, 0).cpu().float()
    
       
        return (images,)

NODE_CLASS_MAPPINGS = {
    "CV_VAE_Load": CV_VAE_Load,
    "CV_VAE_Encode": CV_VAE_Encode,
    "CV_VAE_Decode": CV_VAE_Decode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CV_VAE_Load": "CV_VAE_Load",
    "CV_VAE_Encode": "CV_VAE_Encode",
    "CV_VAE_Decode": "CV_VAE_Decode",
}
