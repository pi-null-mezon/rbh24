import argparse
from templatesgen.tools import CustomDataSet, initialize_onnx_session, make_onnx_inference, normalize_l2
from tqdm import tqdm
import numpy as np
import pickle
import torch


parser = argparse.ArgumentParser("script to generate biometric templates")
parser.add_argument("--images_path", default=None, help="self explained")
parser.add_argument("--output_file", default=f"./templatesgen.pkl", help="where to save templates")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--start_id", default=0, type=int, help="start from id")
parser.add_argument("--max_ids", default=-1, type=int, help="max ids to enroll (-1 - all)")
parser.add_argument("--dataloaders", default=8, type=int, help="dataloaders threads")
args = parser.parse_args()


buffalo = initialize_onnx_session(model_path='./models/buffalo_l/w600k_r50.onnx', use_cuda=torch.cuda.is_available())
antelope = initialize_onnx_session(model_path='./models/antelopev2/glintr100.onnx', use_cuda=torch.cuda.is_available())

dataset = CustomDataSet(images_path=args.images_path, start_id=args.start_id, max_ids=args.max_ids)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         drop_last=False,
                                         pin_memory=True,  # for speedup on CUDA
                                         num_workers=args.dataloaders)

files_list = []
buffalo_features_list = []
antelope_features_list = []

print("FEATURES EXTRACTION - please wait...")
for step, (labels, filenames, samples) in enumerate(tqdm(dataloader)):
    files_list += filenames
    samples = samples.numpy()
    buffalo_features = make_onnx_inference(buffalo, samples)
    antelope_features = make_onnx_inference(antelope, samples)
    if buffalo_features.shape[0] == 1:
        buffalo_features_list += [buffalo_features]
        antelope_features_list += [antelope_features]
    else:
        raise NotImplemented  # buffalo_l model saved with fixed batch size == 1
print("FEATURES EXTRACTION - success")

print("GENUINE FEATURES CONTROL:")
MAX = 5
STEP = 1
print("buffalo:")
buffalo_features = normalize_l2(np.stack(buffalo_features_list[0::STEP][:MAX], axis=1).squeeze(axis=0))
print(np.matmul(buffalo_features, buffalo_features.T))
print("antelope:")
antelope_features = normalize_l2(np.stack(antelope_features_list[0::STEP][:MAX], axis=1).squeeze(axis=0))
print(np.matmul(antelope_features, antelope_features.T))
print("IMPOSTERS FEATURES CONTROL:")
MAX = 5
STEP = 128
print("buffalo:")
buffalo_features = normalize_l2(np.stack(buffalo_features_list[0::STEP][:MAX], axis=1).squeeze(axis=0))
print(np.matmul(buffalo_features, buffalo_features.T))
print("antelope:")
antelope_features = normalize_l2(np.stack(antelope_features_list[0::STEP][:MAX], axis=1).squeeze(axis=0))
print(np.matmul(antelope_features, antelope_features.T))


print(f"SAVING OUTPUT FILE - '{args.output_file}' - please wait...")
with open(args.output_file, 'wb') as o_f:
    pickle.dump({"file": files_list,
                 "buffalo": buffalo_features_list,
                 "antelope": antelope_features_list}, o_f)
print(f"SAVING OUTPUT FILE - '{args.output_file}' - success")



