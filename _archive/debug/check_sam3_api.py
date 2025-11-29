"""
Check what methods are actually available in Sam3Processor.
"""

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
print("Loading SAM 3...")
model = build_sam3_image_model(device='cuda')
processor = Sam3Processor(model)

# Check available methods
print("\nAvailable methods in Sam3Processor:")
methods = [m for m in dir(processor) if not m.startswith('_')]
for method in sorted(methods):
    print(f"  - {method}")

# Check for specific prompt methods
print("\nChecking for prompt methods:")
prompt_methods = [m for m in dir(processor) if 'prompt' in m.lower()]
for method in prompt_methods:
    print(f"  - {method}")

# Check for set methods
print("\nChecking for 'set' methods:")
set_methods = [m for m in dir(processor) if m.startswith('set')]
for method in set_methods:
    print(f"  - {method}")

# Check model architecture
print("\nModel type:", type(model))
print("Processor type:", type(processor))

# Try to access the actual SAM 3 predictor
if hasattr(processor, 'predictor'):
    print("\nProcessor has predictor attribute")
    print("Predictor methods:", [m for m in dir(processor.predictor) if not m.startswith('_')])

if hasattr(processor, 'model'):
    print("\nProcessor has model attribute")
    print("Model type:", type(processor.model))