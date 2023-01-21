import torch
from models.models import Darknet, load_darknet_weights
from utils.torch_utils import select_device, load_classifier, time_synchronized

# set required parameters
input_ = 'yolor_p6.pt'
output = 'yolor_p6.onnx'
model = 'cfg/yolor_p6.cfg'
width = 1280
height = 1280
batch_size = 1

batch_size = 1
device = select_device("cpu", batch_size=batch_size)

# Load model
model = Darknet(model).to(device)
load_darknet_weights(model, input_)

dummy_input = torch.randn((batch_size, 3, width, height),
                          dtype=torch.float32).to(device)

print("Exporting the model using onnx:")
torch.onnx.export(model, dummy_input,
                  output,
                  verbose=False,
                  input_names=['data'], opset_version=11)
