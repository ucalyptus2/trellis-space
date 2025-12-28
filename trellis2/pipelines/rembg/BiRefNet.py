from typing import *
from transformers import AutoModelForImageSegmentation
from transformers.modeling_utils import PreTrainedModel
import torch
from torchvision import transforms
from PIL import Image

# Monkey patch PreTrainedModel to handle missing all_tied_weights_keys
# This is needed for models loaded with trust_remote_code=True in transformers 5.0+
_original_mark_tied_weights = PreTrainedModel.mark_tied_weights_as_initialized

def _patched_mark_tied_weights(self):
    # Add all_tied_weights_keys if it's missing (happens with trust_remote_code models)
    if not hasattr(self, 'all_tied_weights_keys') or self.all_tied_weights_keys is None:
        # Check if there's an alternative attribute that's not None
        if hasattr(self, '_tied_weights_keys') and self._tied_weights_keys is not None:
            self.all_tied_weights_keys = self._tied_weights_keys
        else:
            # Default to empty dict for models that don't use weight tying
            self.all_tied_weights_keys = {}
    return _original_mark_tied_weights(self)

PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied_weights


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # Monkey patch torch.linspace to avoid meta tensor issues
        # This fixes the issue with briaai/RMBG-2.0 model
        original_linspace = torch.linspace

        def safe_linspace(*args, **kwargs):
            # Force CPU device to avoid meta tensor issues
            # Handle both cases: device specified as 'meta' or not specified at all
            if 'device' in kwargs:
                if kwargs['device'] == torch.device('meta') or kwargs['device'] == 'meta':
                    kwargs['device'] = 'cpu'
            else:
                kwargs['device'] = 'cpu'

            # Also ensure we're not using meta dtype
            if 'dtype' not in kwargs:
                kwargs['dtype'] = torch.float32

            return original_linspace(*args, **kwargs)

        torch.linspace = safe_linspace

        try:
            # Load model directly to CPU first to avoid meta tensors
            # Then move to CUDA after initialization is complete
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                device_map='cpu',  # Explicitly load to CPU
            )

            # Convert to float16 and move to CUDA after loading
            self.model = self.model.to(dtype=torch.float16, device='cuda')
            self.model.eval()
        finally:
            # Restore original linspace
            torch.linspace = original_linspace
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    