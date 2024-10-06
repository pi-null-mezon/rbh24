import os
from tools import image2tensor, tensor2image
import torch
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AE = torch.load('./weights/ae_tmp_ae_256lsd_on_vgg11.pth')
AE.eval()

VAE = torch.load('./weights/ae_tmp_ae_256lsd_on_vgg11.pth')
VAE.eval()


def reconstruct(model, bgrs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), swap_red_blue=True):
    with torch.no_grad():
        i_tensor = torch.stack([torch.from_numpy(image2tensor(bgr, mean, std, swap_red_blue)) for bgr in bgrs]).to(device)
        o_tensor = model(i_tensor)
        return [tensor2image(o_tensor[i].cpu().numpy(), mean, std, swap_red_blue) for i in range(o_tensor.shape[0])]


path = './weights'
for filename in [os.path.join(path, f.name) for f in os.scandir(path) if f.is_file() and f.name.lower().rsplit('.',1)[1] in {'jpg', 'png', 'jpeg'}]:
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    cv2.imshow("original", img)
    rec_ae = reconstruct(AE, [img])
    cv2.imshow("AE", cv2.medianBlur(rec_ae[0], 3))
    rec_vae = reconstruct(VAE, [img])
    cv2.imshow("VAE", cv2.medianBlur(rec_vae[0], 3))
    cv2.waitKey(0)
