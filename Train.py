import torch
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler  # provides methods for adjusting the learning rate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

# import DatasetUtils
from FileManager import CleanLogs, SaveModel, CreateModelDir
from DEMDataset import DEMDataset
from Models import Discriminator, Generator


def reverse_mask(x):
    """A function to reverse an image mask"""
    return 1 - x


def discriminator_loss(x, y):
    """ Wasserstein loss function """
    return -(torch.mean(x) - torch.mean(y))


def generator_loss(r, f, m, d):
    # pixel-wise loss
    diff = r - f
    pxlLoss = torch.sqrt(torch.mean(diff*diff))

    # context loss
    """
    w = torch.zeros([Img_Size, Img_Size])  # weight matrix
    for i, j in w:
        if m[i, j] == 0:
            w[i, j] = 0
        else: """

    # perceptual loss
    prcpLoss = torch.log(1 - torch.mean(d))
    # print(f"pxl: {pxlLoss}, prcp: {prcpLoss}")

    return pxlLoss + prcpLoss


# Define Hyper-parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Batch_size = 16
Img_Size = 64  # DatasetUtils.img_size
Img_channels = 1
Z_dim = 100
Num_epochs = 1
Features_disc = 64
Features_gen = 64
Disc_iters = 5
Weight_clip = 0.01  # TODO: check what this is
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-08
Learning_rate = 0.0002

# transformations applied to datasets
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# load the dataset
# DatasetUtils.Create()
# DatasetUtils.Clean(batchSize=Batch_size)
dataset = DEMDataset('lookUpTable.csv', rootDir='LookUp', transform=transforms)
Dataset_size = dataset.__len__()
print("Dataset loaded...")
print(f"Dataset size: {Dataset_size}")

# create dataloader
trainingLoader = DataLoader(dataset=dataset, batch_size=Batch_size, shuffle=True)


# Initialise the two networks
gen = Generator(Z=Z_dim, imgChannels=Img_channels, features=Features_gen).to(device)
print("generator initialised...")
disc = Discriminator(imgChannels=Img_channels, features=Features_disc).to(device)
print("discriminator initialised...")
# initialise weights

# Optimiser Functions
opt_disc = optim.Adam(params=disc.parameters(),
                      lr=Learning_rate,
                      betas=(beta1, beta2),
                      eps=epsilon)

opt_gen = optim.Adam(params=gen.parameters(),
                     lr=Learning_rate,
                     betas=(beta1, beta2),
                     eps=epsilon)

# Define random noise to being training with
fixed_noise = torch.randn(32, Z_dim, 1, 1).to(device)

# Data Visualisation stuff
modelDir, logDir = CreateModelDir()
writer_real = SummaryWriter(logDir + "/real")
writer_fake_masked = SummaryWriter(logDir + "/fake_masked")
writer_fake_raw = SummaryWriter(logDir + "/fake_raw")
writer_d_loss = SummaryWriter(logDir + "/loss_d")
writer_g_loss = SummaryWriter(logDir + "/loss_g")
writer_d_lr = SummaryWriter(logDir + "/lr_d")
writer_g_lr = SummaryWriter(logDir + "/lr_g")
step = 0  # global step value for logs

# set models to training mode
gen.train()
disc.train()
print("ready to train...")

for epoch in range(Num_epochs):
    print(
        "\n==============================================\n"
        f"Epoch [{epoch}/{Num_epochs}] \n"
        "==============================================\n"
    )
    for batch_idx, sample in enumerate(trainingLoader):
        # retrieve ground truth and corresponding mask
        real = sample[0].to(device)
        mask = sample[1].to(device)

        # train discriminator
        for _ in range(Disc_iters):
            noise = torch.rand((Batch_size, Z_dim, 1, 1)).to(device)
            raw, fake = gen(x=noise, m=mask, r=real)

            # send real and fake DEMs to the discriminator
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            # calculate loss
            loss_disc = discriminator_loss(disc_real, disc_fake)
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()
            # disc_lr.step()

            for p in disc.parameters():
                p.data.clamp_(-Weight_clip, Weight_clip)

        # train generator
        gen.zero_grad()
        output = disc(fake).reshape(-1)
        # calculate loss
        loss_gen = generator_loss(real, fake, mask, output)
        loss_gen.backward()
        opt_gen.step()
        print(
            f"{batch_idx}/{len(trainingLoader)} \n"
            f"|| Discriminator Loss: {loss_disc:.4f} \n"
            f"|| Generator Loss: {loss_gen:.4f} \n"

        )

    # display results at the end of each epoch
    print(
        "---------------------------------------- \n"
        f"|| Discriminator Loss: {loss_disc:.4f} \n"
        f"|| Generator Loss: {loss_gen:.4f} \n"
    )

    with torch.no_grad():
        # plot loss functions - not directly useful but can be used to illustrate a point about evaluating GANs
        writer_d_loss.add_scalar('Discriminator Loss', loss_disc, global_step=step)
        writer_g_loss.add_scalar('Generator Loss', loss_gen, global_step=step)
        # pick 16 examples
        img_grid_real = torchvision.utils.make_grid(real[:16])
        img_grid_fake = torchvision.utils.make_grid(fake[:16])
        img_grid_raw = torchvision.utils.make_grid(raw[:16])
        writer_real.add_image("Real", img_grid_real, global_step=step)
        writer_fake_masked.add_image("Fake Masked", img_grid_fake, global_step=step)
        writer_fake_raw.add_image("Fake Raw", img_grid_raw, global_step=step)
    step += 1

    # save model at the end of each epoch
    fileName = "epoch_" + str(epoch) + '.pth'
    checkpoint = {'G': gen,
                  'D': disc,
                  'G_state': gen.state_dict(),
                  'D_state': disc.state_dict(),
                  'Optim_G': opt_gen.state_dict(),
                  'Optim_D': opt_disc.state_dict()}
    SaveModel(checkpoint, modelDir, fileName)
    print(f"Models for epoch {epoch} saved")


# save the final iteration of the model
final = {'G': gen,
         'D': disc,
         'G_state': gen.state_dict(),
         'D_state': disc.state_dict(),
         'Optim_G': opt_gen.state_dict(),
         'Optim_D': opt_disc.state_dict()}
torch.save(final, "model_v1.pth")
