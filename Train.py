import torch
import torch.nn as nn
import torch.optim as optim  # package implementing various optimisation algorithms
import torchvision
from torch.optim import lr_scheduler  # provides methods for adjusting the learning rate
from torch.utils.data import DataLoader  # module for iterating over a dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

import CreateDataset
from DEMDataset import DEMDataset
from Models import Discriminator, Generator


def reverse_mask(x):
    """A function to reverse an image mask"""
    return 1-x


def discriminator_loss(x, y):
    """
    Wasserstein loss function
    :param x: real
    :param y: fake
    :return:
    """
    return -(torch.mean(x) - torch.mean(y))


def generator_loss(r, f, m, d):

    # pixel-wise loss
    i = torch.multiply(f, m)
    t = torch.multiply(r, m)
    loss = torch.nn.MSELoss()
    pxlLoss = loss(i, t)

    # context loss
    """
    w = torch.zeros([Img_Size, Img_Size])  # weight matrix
    for i, j in w:
        if m[i, j] == 0:
            w[i, j] = 0
        else: """

    # perceptual loss
    prcpLoss = torch.log(1-d)
    print(f"pxl loss: {pxlLoss}")

    return pxlLoss


# Define Hyper-parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Learning_rate = 1e-4
Batch_size = 16
Img_Size = CreateDataset.img_size
Img_channels = 1
Z_dim = 100
Num_epochs = 50
Features_disc = 64
Features_gen = 64
Disc_iters = 5
Weight_clip = 0.01  # TODO: check what this is too
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-08

# transformations applied to datasets
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# load the dataset
dataset = DEMDataset('lookUpTable.csv', rootDir='LookUp', transform=transforms)
Dataset_size = dataset.__len__()
print("Dataset loaded...")
print(f"Dataset size: {Dataset_size}")
# split into training and testing sets with an 80/20 ratio
# trainingSet, testingSet = torch.utils.data.random_split(dataset, [int(Dataset_size*8/10), int(Dataset_size*2/10)])
print("Dataset split...")
# create dataloaders for each set
# TODO: look into multiprocessing
trainingLoader = DataLoader(dataset=dataset, batch_size=Batch_size, shuffle=True)
print("training loader created...")
# testingLoader = DataLoader(dataset=testingSet, batch_size=Batch_size, shuffle=True)
print("testing loader created...")


# Initialise the two networks
gen = Generator(Z=Z_dim, imgChannels=Img_channels, features=Features_gen).to(device)
print("generator initialised...")
disc = Discriminator(imgChannels=Img_channels, features=Features_disc).to(device)
print("discriminator initialised...")
# initialise weights

# Optimiser Functions
opt_gen = optim.Adam(params=gen.parameters(),
                     lr=Learning_rate,
                     betas=(beta1, beta2),
                     eps=epsilon)
opt_disc = optim.Adam(params=disc.parameters(),
                      lr=Learning_rate,
                      betas=(beta1, beta2),
                      eps=epsilon)
print("optimisers defined...")

# Define random noise to being training with
fixed_noise = torch.randn(32, Z_dim, 1, 1).to(device)

# Data Visualisation stuff
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
writer_d_loss = SummaryWriter(f"logs/loss_d")
writer_g_loss = SummaryWriter(f"logs/loss_g")
print("Summary writers created...")

step = 0
gen.train()
disc.train()
print("ready to train...")


for epoch in range(Num_epochs):
    for batch_idx, sample in enumerate(trainingLoader):
        # retrieve ground truth and corresponding mask
        real = sample[0].to(device)
        mask = sample[1].to(device)

        # train discriminator
        for _ in range(Disc_iters):
            noise = torch.randn((Batch_size, Z_dim, 1, 1)).to(device)
            generatedDEM = gen(x=noise)

            # apply the reverse mask operation to the generated DEM to create the fake patch
            fakePatch = torch.multiply(generatedDEM, reverse_mask(mask))

            # apply the mask to the real DEM to create the data void
            maskedDEM = torch.multiply(real, mask)

            # combine the fake patch with the masked DEM
            fake = torch.add(maskedDEM, fakePatch)
            # send real and fake DEMs to the discriminator
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            loss_disc = discriminator_loss(disc_real, disc_fake)
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            for p in disc.parameters():
                p.data.clamp_(-Weight_clip, Weight_clip)

        # train generator
        output = disc(fake).reshape(-1)
        loss_gen = generator_loss(r=real, f=fake, m=mask, d=output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # plot loss functions - not directly useful but can be used to illustrate a point about evaluating GANs
        writer_d_loss.add_scalar('Discriminator Loss', loss_disc, global_step=step)
        writer_g_loss.add_scalar('Generator Loss', loss_gen, global_step=step)

        # display results at specified intervals
        if batch_idx % 1 == 0:
            print(
                f"---------------------------------------------- \n"
                f"|| Epoch [{epoch}/{Num_epochs}] -- Batch [{batch_idx}/{len(trainingLoader)}] \n"
                f"|| Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                # pick up to 16 examples
                img_grid_real = torchvision.utils.make_grid(real[:16])
                img_grid_fake = torchvision.utils.make_grid(fake[:16])
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

        step += 1

    # save model at specified epochs
    if epoch+1 % 10 == 0:
        torch.save(gen, 'gen_epoch_{}.pth'.format(epoch))
        print(f"Model {epoch} saved")
