from .qwen_vision_tower import LlavaQwenVisionTower
from .qwen3_vision_tower import LlavaQwen3VisionTower
from .qwen3_5_vision_tower import LlavaQwen3_5VisionTower
from .siglip_encoder import SigLipVisionTower
from llava.utils import rank0_print


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))

    name_lower = vision_tower.lower()
    # Match "qwen3_5" before "qwen3" before "qwen"
    if "qwen3_5" in name_lower or "qwen3.5" in name_lower:
        rank0_print(f"Using LlavaQwen3_5VisionTower: {vision_tower}")
        return LlavaQwen3_5VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "qwen3" in name_lower:
        rank0_print(f"Using LlavaQwen3VisionTower: {vision_tower}")
        return LlavaQwen3VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "qwen" in name_lower:
        rank0_print(f"Using LlavaQwenVisionTower: {vision_tower}")
        return LlavaQwenVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower.lower():
        rank0_print(f"Using SigLipVisionTower: {vision_tower}")
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}. Only Qwen3, Qwen and SigLip vision towers are supported.")
