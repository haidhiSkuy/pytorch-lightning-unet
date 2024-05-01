import torch 
from torch_model import get_model 
from load_params import load_params 



model = get_model()
model = load_params(model, "onnx/brain-epoch=27-val_loss=0.08.ckpt") 

model.eval()
dummy_input = torch.randn(1, 3, 256, 256)

input_names = ["input"]
output_names = ["output"]

torch.onnx.export(
    model,
    dummy_input,
    "brain_unet.onnx",
    verbose=False,
    input_names=input_names,
    output_names=output_names,
    export_params=True,
)