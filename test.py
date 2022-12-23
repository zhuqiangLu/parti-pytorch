from parti_pytorch import VitVQGanVAE, VQGanVAETrainer
import torch


vit_vae = VitVQGanVAE(
    dim = 256,               # dimensions
    image_size = 256,        # target image size
    patch_size = 16,         # size of the patches in the image attending to each other
    num_layers = 3           # number of layers
).cuda()

vit_vae.load_state_dict(torch.load(r'./ckpts/vae.6000.pt'))

trainer = VQGanVAETrainer(
    vit_vae,
    folder = r'I:\share\zhlu6105\dataset\MONO\vitdata',
    num_train_steps = 100000,
    lr = 3e-5,
    batch_size = 8,
    grad_accum_every = 8,
    amp = True
)

trainer.train()