import os
import sys
import cv2
import numpy
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from easydict import EasyDict as edict
from tools import CustomDataSet, Averagemeter, Speedometer, print_one_line, model_size_mb, tensor2image
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from nn.customloss import CustomLossForAutoencoder as CustomLoss
from nn import neuralnet
from nn import perceptloss
from nn.discriminator import get_discriminator_network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- TRAIN CONFIG -----------

cfg = edict()
cfg.mean = [0.5, 0.5, 0.5]
cfg.std = [0.5, 0.5, 0.5]
cfg.swap_red_blue = True  # torchvision default
cfg.crop_size = (112, 112)  # (128, 128)         # treat as (width, height)
cfg.batch_size = 200
cfg.num_epochs = 10
cfg.lr_scheduler = "Cosine"  # "StepPlateau"
cfg.perception_net_name = 'fr'  # 'vgg11'
cfg.latent_space_dims = 512  # as features size for insightface/buffalo_l is
cfg.max_lr = 0.001
cfg.min_lr = 0.00001
cfg.max_grad_norm = 10.0
cfg.augment = False
cfg.normalize_templates = True
cfg.max_batches_per_train_epoch = -1  # -1 - use all available batches
cfg.visualize_control_samples = 9  # how many tst samples will be used for online visual control
cfg.visualize_each_step = 32 * 32  # 1024
cfg.perceptual_loss_weight = 11.0
cfg.laplace_loss_weight = 1.0
cfg.mse_loss_weight = 1.0
cfg.use_discriminator = False
cfg.discr_weight = 0.01
cfg.run_name = 'wo_discr_wo_pixel_loss'
cfg.model_name = f"buffalo_decoder_on_{cfg.perception_net_name}_{cfg.run_name}"
cfg.save_powers = {10 ** x for x in [2, 3, 4, 5]}
# -------- SET PATH TO LOCAL DATA ---------

local_data_path = f"../data"
photos_path = f"/mount/hdd1/recognition/train/glint360k/images"
train_templates_paths = [
    f"{local_data_path}/templatesgen_glint_10K_20K.pkl",
    f"{local_data_path}/templatesgen_glint_20K_30K.pkl",
    f"{local_data_path}/templatesgen_glint_30K_40K.pkl",
    f"{local_data_path}/templatesgen_glint_40K_50K.pkl",
    f"{local_data_path}/templatesgen_glint_50K_60K.pkl",
    f"{local_data_path}/templatesgen_glint_60K_70K.pkl",
    f"{local_data_path}/templatesgen_glint_70K_80K.pkl",
    f"{local_data_path}/templatesgen_glint_80K_90K.pkl",
    f"{local_data_path}/templatesgen_glint_90K_100K.pkl",
    f"{local_data_path}/templatesgen_glint_100K_140K.pkl",
    f"{local_data_path}/templatesgen_glint_140K_180K.pkl",
    f"{local_data_path}/templatesgen_glint_180K_200K.pkl",
    f"{local_data_path}/templatesgen_glint_200K_210K.pkl",
    f"{local_data_path}/templatesgen_glint_210K_220K.pkl",
    f"{local_data_path}/templatesgen_glint_220K_240K.pkl",
]
test_templates_paths = [
    f"{local_data_path}/templatesgen_glint_0_10K.pkl",
]

# ---------- SETUP LOGGING ------------

for key in cfg:
    print(f" - {key}: {cfg[key]}")
writer = SummaryWriter()

# --------------------------- NEURAL NETS, LOSSES, OPTIMIZER & LR SCHEDULER

model = neuralnet.ConvFaceDecoder(latent_dim=cfg.latent_space_dims)
model = model.to(device)
print(f" - model size: {model_size_mb(model):.3f} MB")

perceptor = perceptloss.get_perceptual_loss_network(cfg.perception_net_name).to(device)
print(f" - perceptor ({cfg.perception_net_name}) size: {model_size_mb(perceptor):.3f} MB")

# Loss and optimizer
loss_fn = CustomLoss(mse_k=cfg.mse_loss_weight, ll_k=cfg.laplace_loss_weight, pl_k=cfg.perceptual_loss_weight)
optimizer = optim.Adam(model.parameters(), lr=cfg.max_lr)

# Discriminator
if cfg.use_discriminator:
    loss_fn_bce = torch.nn.BCEWithLogitsLoss()
    discriminator = get_discriminator_network('tiny')
    discriminator.cuda()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.max_lr)

if cfg.lr_scheduler == "StepPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=2,
                                                           factor=0.2,
                                                           min_lr=cfg.min_lr,
                                                           verbose=True)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=cfg.num_epochs,
                                                                     T_mult=1,
                                                                     eta_min=cfg.min_lr,
                                                                     verbose=True)

# ---------------------------- DATASETS

print("Train dataset:")
train_dataset = CustomDataSet(templates_paths=train_templates_paths,
                              photos_path=photos_path,
                              size=cfg.crop_size,
                              do_aug=cfg.augment,
                              mean=cfg.mean,
                              std=cfg.std,
                              swap_reb_blue=cfg.swap_red_blue,
                              normalize_templates=cfg.normalize_templates)
print(f"  - unique samples: {len(train_dataset.templates)}")
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               drop_last=True,
                                               pin_memory=True)

print("Test dataset:")
test_dataset = CustomDataSet(templates_paths=test_templates_paths,
                             photos_path=photos_path,
                             size=cfg.crop_size,
                             do_aug=False,
                             mean=cfg.mean,
                             std=cfg.std,
                             swap_reb_blue=cfg.swap_red_blue,
                             normalize_templates=cfg.normalize_templates)
print(f"  - unique samples: {len(test_dataset.templates)}")
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              drop_last=True,
                                              pin_memory=True)

# ---------------------------- METRICS

metrics = {
    'train': {'loss': float('inf')},
    'test': {'loss': float('inf')}
}


def update_metrics(mode, epoch, loss):
    assert not numpy.isnan(loss), "Loss is nan! Terminating train..."
    if mode == 'train':
        print()
    if not os.path.exists('./weights'):
        os.makedirs('./weights')
    writer.add_scalar(f"loss/{mode}", loss, epoch)
    if loss < metrics[mode]['loss']:
        metrics[mode]['loss'] = loss
        print(f" - loss:  {loss:.5f} - improvement")
        if mode == 'test':
            torch.save(model, f"./weights/_tmp_{cfg.model_name}.pth")
    else:
        print(f" - loss:  {loss:.5f}")


loss_avgm = Averagemeter()
speedometer = Speedometer()
scaler = amp.grad_scaler.GradScaler()


# ---------------------------- TRAIN LOGIC

def train(epoch, dataloader):
    print("TRAIN:")
    loss_avgm.reset()
    speedometer.reset()
    model.train()
    running_loss = 0
    samples_enrolled = 0
    for batch_idx, (templates, photos) in enumerate(dataloader):
        if batch_idx == cfg.max_batches_per_train_epoch:
            break
        templates = templates.to(device)
        photos = photos.to(device)
        outputs = model(templates)  # [:, :, :cfg.crop_size[0], : cfg.crop_size[1]]
        opf = perceptor(photos)
        rpf = perceptor(outputs)

        loss = loss_fn(outputs, photos, opf, rpf)
        # Discriminator optimization
        if cfg.use_discriminator:
            optimizer_d.zero_grad()
            discr_pred = discriminator(torch.cat([photos, outputs.detach()]))
            discr_target = torch.tensor([1. for i in range(len(photos))] + [0. for i in range(len(outputs))]).unsqueeze(
                1).cuda()
            discr_loss = loss_fn_bce(discr_pred, discr_target)
            discr_loss.backward()
            optimizer_d.step()

            # Generator optimization
            discr_pred = discriminator(outputs)
            discr_target = torch.tensor([0. for i in range(len(outputs))]).unsqueeze(1).cuda()
            discr_loss = loss_fn_bce(discr_pred, discr_target)
            loss -= (cfg.discr_weight * discr_loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            running_loss += loss.item()
            loss_avgm.update(loss.item())
            samples_enrolled += templates.size(0)
            speedometer.update(templates.size(0))
            if (batch_idx % cfg.visualize_each_step == 0) or (
                    (((batch_idx + 1) * cfg.batch_size // 10) in cfg.save_powers) and epoch == 0):
                model.eval()
                for i, sample in enumerate(visual_control_samples):
                    probe = model(sample[0])  # [:, :, :cfg.crop_size[0], : cfg.crop_size[1]]
                    orig = tensor2image(sample[1].cpu().numpy(),
                                        mean=cfg.mean,
                                        std=cfg.std,
                                        swap_red_blue=cfg.swap_red_blue)
                    rec = tensor2image(probe.squeeze(0).cpu().numpy(),
                                       mean=cfg.mean,
                                       std=cfg.std,
                                       swap_red_blue=cfg.swap_red_blue)
                    canvas = np.zeros(shape=(orig.shape[0], orig.shape[1] + rec.shape[1], orig.shape[2]),
                                      dtype=np.uint8)
                    canvas[0:orig.shape[0], 0:orig.shape[1]] = orig
                    canvas[0:rec.shape[0], orig.shape[1]:orig.shape[1] + rec.shape[1]] = rec
                    os.makedirs(f"results_{cfg.run_name}", exist_ok=True)
                    cv2.imwrite(f"results_{cfg.run_name}/original_vs_reconstructed#{i}.png", canvas)
                y = ((batch_idx + 1) * cfg.batch_size // 10) in cfg.save_powers
                postfix = str((batch_idx + 1) * cfg.batch_size) if y else 'last'
                torch.save(model, f"./weights/_tmp_{cfg.model_name}_{postfix}.pth")
                model.train()
        print_one_line(
            f'Epoch {epoch} >> loss {loss_avgm.val:.5f} | '
            f'{samples_enrolled}/{len(train_dataset)} ~ '
            f'{100 * samples_enrolled / len(train_dataset):.1f} % | '
            f'{speedometer.speed():.0f} samples / s '
        )
    update_metrics('train', epoch, running_loss / batch_idx)


# ---------------------------- TEST LOGIC


def test(epoch, dataloader):
    print("TEST:")
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for batch_idx, (templates, photos) in enumerate(tqdm(dataloader, file=sys.stdout)):
            templates = templates.to(device)
            photos = photos.to(device)
            outputs = model(templates)  # [:, :, :cfg.crop_size[0], : cfg.crop_size[1]]
            opf = perceptor(photos)
            rpf = perceptor(outputs)
            loss = loss_fn(outputs, photos, opf, rpf)
            running_loss += loss.item()
    update_metrics('test', epoch, running_loss / len(dataloader))
    if cfg.lr_scheduler == "StepPlateau":
        scheduler.step(metrics['test']['loss'])
    else:
        scheduler.step()
    print("\n")


# ----------------------------


print("SELECTING SAMPLES FOR VISUAL CONTROL - please wait...")
visual_control_samples = []
for batch_idx, (templates, photos) in enumerate(test_dataloader):
    templates = templates.to(device)
    photos = photos.to(device)
    if len(visual_control_samples) < cfg.visualize_control_samples:
        visual_control_samples.append((templates[0], photos[0]))
    else:
        break
print("SELECTING SAMPLES FOR VISUAL CONTROL - success")
for epoch in range(cfg.num_epochs):
    train(epoch, train_dataloader)
    test(epoch, test_dataloader)
cv2.waitKey(-1)
