# find_perfect_resolution_live_simple.py
# Version 0.14 - Avec support MASK
# Auteur: ashtar1984 + ChatGPT

import math
import torch
import numpy as np
from PIL import Image

class FindPerfectResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "desired_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "desired_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "divisible_by": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  # Nouvelle entrée MASK optionnelle
                "upscale": ("BOOLEAN", {"default": False}),
                "upscale_method": (["lanczos", "bilinear", "bicubic", "nearest"], {"default": "lanczos"}),
                "small_image_mode": (["none", "crop", "pad"], {"default": "none"}),
                "pad_color": ("STRING", {"default": "#000000"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",  # important pour l'affichage live
            }
        }

    RETURN_TYPES = ("INT", "INT", "IMAGE", "MASK", "STRING") # Ajout de MASK
    RETURN_NAMES = ("width", "height", "IMAGE", "mask_output", "resolution_info") # Ajout de mask_output
    FUNCTION = "calculate"
    CATEGORY = "utils"

    def calculate(self, image, desired_width, desired_height, divisible_by,
                  mask=None,  # Ajout de l'argument mask
                  upscale=False, upscale_method="lanczos",
                  small_image_mode="none", pad_color="#000000",
                  unique_id=None):

        _, orig_h, orig_w, _ = image.shape
        aspect_ratio = orig_w / orig_h

        # --- Auto calcul si 0 ---
        if desired_width == 0 and desired_height == 0:
            raise ValueError("desired_width et desired_height ne peuvent PAS être tous les deux à 0.")
        if desired_width == 0:
            desired_width = int(desired_height * aspect_ratio)
        if desired_height == 0:
            desired_height = int(desired_width / aspect_ratio)

        # --- Calcul résolution divisible ---
        num_pixels = desired_width * desired_height
        h_float = math.sqrt((num_pixels * orig_h) / orig_w)
        new_h = max(divisible_by, round(h_float / divisible_by) * divisible_by)
        new_w = max(divisible_by, round((aspect_ratio * h_float) / divisible_by) * divisible_by)

        method_map = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST,
        }
        resize_method = method_map.get(upscale_method, Image.LANCZOS)
        
        # --- Redimensionnement de l'IMAGE ---
        results = []
        for i in range(image.shape[0]):
            pil_img = Image.fromarray((image[i].cpu().numpy() * 255).astype(np.uint8))

            is_upscale = new_w > orig_w or new_h > orig_h
            do_resize = not is_upscale or (is_upscale and upscale)

            if do_resize:
                if small_image_mode != "none" and (pil_img.width < new_w or pil_img.height < new_h):
                    target_ar = new_w / new_h
                    img_ar = pil_img.width / pil_img.height
                    
                    if small_image_mode == "crop":
                        if img_ar > target_ar:
                            tmp_h = new_h
                            tmp_w = int(tmp_h * img_ar)
                        else:
                            tmp_w = new_w
                            tmp_h = int(tmp_w / img_ar)
                        pil_img = pil_img.resize((tmp_w, tmp_h), resize_method)
                        left = (pil_img.width - new_w) // 2
                        top = (pil_img.height - new_h) // 2
                        pil_img = pil_img.crop((left, top, left + new_w, top + new_h))
                    
                    elif small_image_mode == "pad":
                        pil_img.thumbnail((new_w, new_h), resize_method)
                        bg = Image.new("RGB", (new_w, new_h), self._hex_to_rgb(pad_color))
                        offset = ((new_w - pil_img.width) // 2, (new_h - pil_img.height) // 2)
                        bg.paste(pil_img, offset)
                        pil_img = bg
                else:
                    pil_img = pil_img.resize((new_w, new_h), resize_method)

            img_np = np.array(pil_img).astype(np.float32) / 255.0
            results.append(img_np)

        image_out = torch.from_numpy(np.stack(results)).to(image.device)
        
        # --- Traitement du MASK ---
        mask_out = None
        if mask is not None:
            # ComfyUI MASK est de forme (B, H, W)
            mask_resized_list = []
            for i in range(mask.shape[0]):
                # Convertir le tenseur (H, W) en PIL Image (mode 'L' pour niveaux de gris)
                mask_np = (mask[i].cpu().numpy() * 255).astype(np.uint8)
                pil_mask = Image.fromarray(mask_np, mode='L')

                # Appliquer le même redimensionnement/recadrage que pour l'image
                if do_resize:
                    if small_image_mode != "none" and (pil_mask.width < new_w or pil_mask.height < new_h):
                        target_ar = new_w / new_h
                        img_ar = pil_mask.width / pil_mask.height # Utiliser la taille originale du masque (qui devrait être la même que l'image)
                        
                        if small_image_mode == "crop":
                            if img_ar > target_ar:
                                tmp_h = new_h
                                tmp_w = int(tmp_h * img_ar)
                            else:
                                tmp_w = new_w
                                tmp_h = int(tmp_w / img_ar)
                            
                            # Redimensionnement avant le recadrage.
                            # Utilisez Image.NEAREST pour les masques afin de préserver les bords nets
                            # sauf si l'utilisateur choisit explicitement un mode d'upscale plus doux.
                            mask_resize_method = Image.NEAREST if upscale_method == "nearest" else resize_method
                            pil_mask = pil_mask.resize((tmp_w, tmp_h), mask_resize_method)

                            # Recadrage
                            left = (pil_mask.width - new_w) // 2
                            top = (pil_mask.height - new_h) // 2
                            pil_mask = pil_mask.crop((left, top, left + new_w, top + new_h))
                        
                        elif small_image_mode == "pad":
                            pil_mask.thumbnail((new_w, new_h), Image.NEAREST) # NEAREST pour le masque
                            bg = Image.new("L", (new_w, new_h), 0) # Remplissage noir (0)
                            offset = ((new_w - pil_mask.width) // 2, (new_h - pil_mask.height) // 2)
                            bg.paste(pil_mask, offset)
                            pil_mask = bg
                    else:
                         # Redimensionnement simple
                        pil_mask = pil_mask.resize((new_w, new_h), resize_method) 
                        
                # Convertir la PIL Image en tenseur ComfyUI MASK (H, W)
                mask_np_out = np.array(pil_mask).astype(np.float32) / 255.0
                mask_resized_list.append(mask_np_out)
            
            # Empiler les masques pour obtenir (B, H, W)
            mask_out = torch.from_numpy(np.stack(mask_resized_list)).to(image.device)
            # S'assurer que les masques ont la bonne dimension (pas de canal de couleur)
            if len(mask_out.shape) == 4:
                 mask_out = mask_out[..., 0] 


        # --- Infos pour affichage ---
        approx_bytes = new_w * new_h * 3 # Estimer avec 3 canaux pour l'image
        approx_mb = approx_bytes / (1024*1024)
        resolution_info = f"{new_w}x{new_h} | {approx_mb:.2f}MB | {new_w*new_h:,} pixels"

        # --- Affichage live sous le node ---
        if unique_id:
            try:
                from server import PromptServer
                memory_size_mb = (image_out.numel() * image_out.element_size()) / (1024*1024)
                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td>"
                    f"<td><b>{new_w}</b> x <b>{new_h}</b> | {memory_size_mb:.2f}MB | {new_w*new_h:,} pixels</td></tr>",
                    unique_id
                )
            except Exception:
                pass
        
        # Retourner l'IMAGE et le MASK (qui peut être None si non fourni)
        return int(new_w), int(new_h), image_out, mask_out, resolution_info

    def _hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) if len(hex_color) == 6 else (0, 0, 0)
