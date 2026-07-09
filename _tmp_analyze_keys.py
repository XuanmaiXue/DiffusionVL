import json
from collections import Counter

p = "C:/Users/xuan/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3/model.safetensors.index.json"
with open(p) as f:
    d = json.load(f)
wm = d["weight_map"]
keys = list(wm.keys())
print("total keys:", len(keys))
print("shards:", sorted(set(wm.values())))
print()

prefixes = Counter()
for k in keys:
    parts = k.split(".")
    prefixes[".".join(parts[:2])] += 1
print("=== top-2-level prefix counts ===")
for pfx, c in prefixes.most_common():
    print(f"  {pfx:42s} {c}")
print()

print("=== visual.* sample (first 12) ===")
vk = [k for k in keys if k.startswith("visual.")]
for k in vk[:12]:
    print(" ", k)
print(f"  ... ({len(vk)} total visual keys)")
print()

print("=== merger keys ===")
for k in keys:
    if "merger" in k:
        print(" ", k)
print()

print("=== lm_head / embed / norm ===")
for k in keys:
    if k.startswith("lm_head") or k.startswith("model.embed") or k.startswith("model.norm"):
        print(" ", k)

print()
print("=== language layer sample (layer 0) ===")
for k in sorted(keys):
    if k.startswith("model.layers.0."):
        print(" ", k)
