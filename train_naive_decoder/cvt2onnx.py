import torch
import numpy
import onnx
import time
import onnxruntime
from tools import torch2numpy, initialize_onnx_session, make_onnx_inference
from easydict import EasyDict as edict

# --------- CONVERSION CONFIG -----------
cfg = edict()
cfg.template_shape = (1, 512)
cfg.target_onnx_opset = 11
cfg.model_weights = './weights/_tmp_buffalo_decoder_large_on_vgg11.pth'
# ------------------------------------------------------------------------------------

model = torch.load(cfg.model_weights, map_location=torch.device('cpu')).eval()
dummy_input = torch.randn(cfg.template_shape)

target_name_onnx = cfg.model_weights.rsplit('.pth', 1)[0] + ".onnx"
torch.onnx.export(model,
                  dummy_input,
                  target_name_onnx,
                  verbose=False,
                  export_params=True,
                  opset_version=cfg.target_onnx_opset,  # the ONNX version to export the model to
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # variable length axes
                  )
onnx_model = onnx.load(target_name_onnx)
onnx.checker.check_model(onnx_model)
session = initialize_onnx_session(target_name_onnx, use_cuda=False)
prediction_onnx = make_onnx_inference(session=session, input_data=torch2numpy(dummy_input))
prediction_torch = model(dummy_input)
numpy.testing.assert_allclose(torch2numpy(prediction_torch), prediction_onnx, rtol=1e-03, atol=1e-05)
print("Exported onnx model has been tested with onnxruntime, and the result looks similar to torch :)")

print("Performance benchmark:")
repeats = 100
warmup = 10
print(f" > repeats: {repeats}")
print(f" > warmup: {warmup}")
t = []
with torch.no_grad():
    for i in range(repeats + warmup):
        x = torch.randn(cfg.template_shape)
        t0 = time.perf_counter()
        model(x)
        if i > warmup:
            t.append(time.perf_counter() - t0)
t = numpy.asarray(t)
print(f" > pytorch runtime:")
print(f"       - min: {numpy.min(t).item()*1000.0:.1f} ms")
print(f"       - avg: {numpy.average(t).item()*1000.0:.1f} ms")
print(f"       - med: {numpy.median(t).item()*1000.0:.1f} ms")
print(f"       - max: {numpy.max(t).item()*1000.0:.1f} ms")

t = []
for i in range(repeats + warmup):
    x = torch2numpy(torch.randn(cfg.template_shape))
    t0 = time.perf_counter()
    make_onnx_inference(session=session, input_data=x)
    if i > warmup:
        t.append(time.perf_counter() - t0)
t = numpy.asarray(t)
print(f" > onnx runtime:")
print(f"       - min: {numpy.min(t).item()*1000.0:.1f} ms")
print(f"       - avg: {numpy.average(t).item()*1000.0:.1f} ms")
print(f"       - med: {numpy.median(t).item()*1000.0:.1f} ms")
print(f"       - max: {numpy.max(t).item()*1000.0:.1f} ms")