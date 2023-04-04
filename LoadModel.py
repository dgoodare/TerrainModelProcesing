import os
import random

import torch
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler  # provides methods for adjusting the learning rate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

import DatasetUtils
from FileManager import CreateModelDir
from DEMDataset import DEMDataset
from Models import Discriminator, Generator

# Define Hyper-parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Batch_size = 16
Img_Size = DatasetUtils.img_size
Img_channels = 1
Z_dim = 100
Num_epochs = 10
Features_disc = 64
Features_gen = 64
Disc_iters = 5
Weight_clip = 0.01  # TODO: check what this is
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-08


def LoadOld(g, d):
    gen = Generator(Z=Z_dim, imgChannels=Img_channels, features=Features_gen)
    disc = Discriminator(imgChannels=Img_channels, features=Features_disc)

    gen.load_state_dict(torch.load(g))
    gen = torch.nn.DataParallel(gen)
    disc.load_state_dict(torch.load(d))
    disc = torch.nn.DataParallel(disc)

    # Optimiser Functions
    opt_disc = optim.Adam(params=disc.parameters(),
                          betas=(0.9, 0.999),
                          eps=1e-08)

    opt_gen = optim.Adam(params=gen.parameters(),
                         betas=(0.9, 0.999),
                         eps=1e-08)

    Train(gen, disc, opt_gen, opt_disc)


def Load(file, mode=''):
    checkpoint = torch.load(file)
    print(checkpoint.keys())
    gen = checkpoint["G"]
    gen.load_state_dict(checkpoint["G_state"])
    disc = checkpoint["D"]
    disc.load_state_dict(checkpoint["D_state"])

    # Optimiser Functions
    opt_gen = optim.Adam(params=gen.parameters(),
                         betas=(0.9, 0.999),
                         eps=1e-08)
    opt_gen.load_state_dict(checkpoint['Optim_G'])

    opt_disc = optim.Adam(params=disc.parameters(),
                          betas=(0.9, 0.999),
                          eps=1e-08)
    opt_disc.load_state_dict(checkpoint['Optim_D'])

    if mode == 'Train':
        Train(gen, disc, opt_gen, opt_disc)
    elif mode == 'Eval':
        Generate(gen, '', 100)
    else:
        print(f"{mode} is not a valid mode")


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


def Train(gen, disc, opt_gen, opt_disc):
    gen.to(device)
    disc.to(device)

    dataset = DEMDataset('lookUpTable.csv', rootDir='LookUp', transform=transforms)
    Dataset_size = dataset.__len__()
    print("Dataset loaded...")
    print(f"Dataset size: {Dataset_size}")
    # split into training and testing sets with an 80/20 ratio
    # trainingSet, testingSet = torch.utils.data.random_split(dataset, [int(Dataset_size*8/10), int(Dataset_size*2/10)])
    # create dataloaders for each set
    # TODO: look into multiprocessing
    trainingLoader = DataLoader(dataset=dataset, batch_size=Batch_size, shuffle=True)
    # testingLoader = DataLoader(dataset=testingSet, batch_size=Batch_size, shuffle=True)
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

    # create a directory to save trained models created in this training run

    step = 0
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
            # gen_lr.step()

            # plot loss functions - not directly useful but can be used to illustrate a point about evaluating GANs
            writer_d_loss.add_scalar('Discriminator Loss', loss_disc, global_step=step)
            writer_g_loss.add_scalar('Generator Loss', loss_gen, global_step=step)
            # plot learning rates
            # writer_d_lr.add_scalar('Discriminator Learning Rate', disc_lr.get_last_lr()[0], global_step=step)
            # writer_g_lr.add_scalar('Generator Learning Rate', gen_lr.get_last_lr()[0], global_step=step)

            # display results at specified intervals
            if batch_idx % 1 == 0:
                print(
                    f"---------------------------------------------- \n"
                    f"-> Batch [{batch_idx}/{len(trainingLoader)}]\n"
                    f"|| Discriminator Loss: {loss_disc:.4f} \n"
                    f"|| Generator Loss: {loss_gen:.4f} \n"
                )

                with torch.no_grad():
                    # pick up to 16 examples
                    img_grid_real = torchvision.utils.make_grid(real[:16])
                    img_grid_fake = torchvision.utils.make_grid(fake[:16])
                    img_grid_raw = torchvision.utils.make_grid(raw[:16])
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake_masked.add_image("Fake Masked", img_grid_fake, global_step=step)
                    writer_fake_raw.add_image("Fake Raw", img_grid_raw, global_step=step)

            step += 1


def Generate(model, outputDir, numSamples, maskDir='Evaluation/outputMasks', inputDir='Evaluation/outputSlices'):
    model.eval()
    masks = os.listdir(maskDir)
    counter = 0

    for file in os.listdir(inputDir):
        # select a random mask to apply to the sample
        idx = random.randrange(len(masks))
        mask = torch.load(masks[idx])

        # load sample
        sample = torch.load(file)

        with torch.no_grad():
            # apply in-filling model
            output = model(sample, mask)
            # save output to file
            path = outputDir + '/' + str(counter) + '_sample.pt'
            torch.save(output, path)

        counter += 1
        if counter == numSamples:
            break


Load('models/27-03-2023_20-37/epoch_9.pth')
