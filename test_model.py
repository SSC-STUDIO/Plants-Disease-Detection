import timm
import torch
print("Creating EfficientNetV2-S model (pretrained=True)...", flush=True)
try:
    m = timm.create_model("efficientnetv2_s", pretrained=True, num_classes=59)
    print("Model created successfully with pretrained weights!", flush=True)
    print(f"Model parameters: {sum(p.numel() for p in m.parameters()):,}", flush=True)
except Exception as e:
    print(f"Failed with pretrained=True: {e}", flush=True)
    print("Trying without pretrained weights...", flush=True)
    m = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=59)
    print("Model created successfully WITHOUT pretrained weights.", flush=True)
    print(f"Model parameters: {sum(p.numel() for p in m.parameters()):,}", flush=True)

# Test GPU
if torch.cuda.is_available():
    print("Moving model to GPU...", flush=True)
    m = m.cuda()
    print("Model on GPU successfully!", flush=True)
else:
    print("CUDA not available - will train on CPU", flush=True)

print("Test complete!", flush=True)
