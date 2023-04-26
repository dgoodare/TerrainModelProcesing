import os
import random
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler  # provides methods for adjusting the learning rate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

import DatasetUtils
from FileManager import CreateModelDir, SaveModel
from DEMDataset import DEMDataset
from Models import Discriminator, Generator

from skimage.io import imshow

# Define Hyper-parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Batch_size = 16
Img_Size = 64
Img_channels = 1
Z_dim = 100
Features_disc = 64
Features_gen = 64
Disc_iters = 5
Weight_clip = 0.01  # TODO: check what this is
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-08


def Load(file, outDir, mode='', epochs=10):
    checkpoint = torch.load(file)
    print(checkpoint.keys())

    if mode == 'Train':
        Train(checkpoint)
    elif mode == 'Eval':
        os.mkdir(outDir)
        Generate(checkpoint, outputDir=outDir, numSamples=100)
    else:
        print(f"{mode} is not a valid mode")


def reverse_mask(x):
    """A function to reverse an image mask"""
    return 1 - x


def discriminator_loss(x, y):
    """ Wasserstein loss function """
    return -(torch.mean(x) - torch.mean(y))


def generator_loss(r, f, m, w, d):
    # pixel-wise loss
    diff = (r - f) * (1-m)
    pxlLoss = torch.mean(torch.abs(diff))

    # context loss
    ctxLoss = torch.mean(torch.abs(w * (f - r)))

    # perceptual loss
    prcLoss = torch.log(1 - torch.mean(d))
    # print(f"pxl: {pxlLoss}, ctx: {ctxLoss} prc: {prcLoss}")

    return pxlLoss, ctxLoss, prcLoss


def gradient_penalty(model, r, f):
    B, C, H, W = r.shape  # batch size, channels, height, width
    alpha = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # interpolate the set of images
    interpolated = r * alpha + f * (1 - alpha)
    # calculate discriminator scores
    disc_scores = model(interpolated)

    # calculate gradients
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=disc_scores,
        grad_outputs=torch.ones_like(disc_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp


def Train(file, Num_epochs=10):
    checkpoint = torch.load(file)

    gen = Generator(Z=Z_dim, imgChannels=Img_channels, features=Features_gen).to(device)
    gen.load_state_dict(checkpoint["G_state"])
    disc = Discriminator(imgChannels=Img_channels, features=Features_disc).to(device)
    disc.load_state_dict(checkpoint["D_state"])

    Learning_rate = 0.000002
    opt_disc = optim.Adam(params=disc.parameters(),
                          lr=Learning_rate,
                          betas=(beta1, beta2),
                          eps=epsilon)

    opt_gen = optim.Adam(params=gen.parameters(),
                         lr=Learning_rate,
                         betas=(beta1, beta2),
                         eps=epsilon)
    opt_disc.load_state_dict(checkpoint['Optim_D'])
    opt_gen.load_state_dict(checkpoint['Optim_G'])

    dataset = DEMDataset('lookUpTable.csv', rootDir='LookUp', transform=transforms)
    Dataset_size = dataset.__len__()
    print("Dataset loaded...")
    print(f"Dataset size: {Dataset_size}")

    # create dataloader
    trainingLoader = DataLoader(dataset=dataset, batch_size=Batch_size, shuffle=True)

    # Define random noise to being training with
    fixed_noise = torch.rand((Batch_size, Z_dim, 1, 1)).to(device)

    # Data Visualisation stuff
    modelDir, logDir = CreateModelDir()
    writer_real = SummaryWriter(logDir + "/real")
    writer_fake_masked = SummaryWriter(logDir + "/fake_masked")
    writer_fake_raw = SummaryWriter(logDir + "/fake_raw")
    writer_d_loss = SummaryWriter(logDir + "/loss_d")
    writer_g_loss = SummaryWriter(logDir + "/loss_g")
    writer_pxl = SummaryWriter(logDir + "/pxl")
    writer_prc = SummaryWriter(logDir + "/prc")
    writer_ctx = SummaryWriter(logDir + "/ctx")

    step = 0  # global step value for logs

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # set models to training mode
    gen.train()
    disc.train()
    print("ready to train...")

    for epoch in range(Num_epochs):
        print(
            "\n==============================================\n"
            f"Epoch [{epoch + 1}/{Num_epochs}] \n"
            "==============================================\n"
        )
        for batch_idx, sample in enumerate(trainingLoader):
            # retrieve ground truth and corresponding mask
            real = sample[0].to(device)
            mask = sample[1].to(device)
            weightMatrix = sample[2].to(device)

            #######################
            # Train Discriminator #
            #######################
            disc.zero_grad()
            b_size = real.shape[0]
            # train using only real data
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # forward pass through discriminator
            output = disc(real).view(-1)

            disc_loss_real = torch.mean(output)  # calculate loss for the real batch
            # backward pass through discriminator
            d_x = output.mean().item()

            # train using only fake data
            noise = torch.rand((Batch_size, Z_dim, 1, 1)).to(device)
            raw = gen(noise)
            label.fill_(fake_label)
            # classify fake data
            output = disc(raw.detach()).view(-1)
            disc_loss_fake = torch.mean(output)
            d_g1 = output.mean().item()

            # calculate gradient penalty
            gp = gradient_penalty(disc, r=real, f=raw)

            # compute error as sum of real and fake batches
            disc_loss = -disc_loss_real + disc_loss_fake + 10 * gp
            disc_loss.backward(retain_graph=True)
            opt_disc.step()

            ###################
            # Train Generator #
            ###################
            gen.zero_grad()
            label.fill_(real_label)  # generator uses inverse labels
            # perform another forward pass through discriminator
            fake = (mask * real) + ((1 - mask) * raw)
            fake_output = disc(fake).view(-1)

            pxlLoss, ctxLoss, prcLoss = generator_loss(r=real, f=fake, m=mask, w=weightMatrix, d=fake_output)
            gen_loss = pxlLoss + ctxLoss  # + prcLoss
            d_g2 = fake_output.mean().item()
            gen_loss.backward()

            opt_gen.step()

            if batch_idx % 10 == 0:
                print(f"{batch_idx}/{len(trainingLoader)}")

                print('Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (disc_loss.item(), gen_loss.item(), d_x, d_g1, d_g2))
                print(f"pxl: {pxlLoss}, ctx: {ctxLoss} prc: {prcLoss}")

                with torch.no_grad():
                    raw = gen(x=fixed_noise)
                    fake = (mask * real) + ((1 - mask) * raw)

                    # plot loss functions - not directly useful but can be used to illustrate a point about evaluating GANs
                    writer_d_loss.add_scalar('Discriminator Loss', disc_loss, global_step=step)
                    writer_g_loss.add_scalar('Generator Loss', gen_loss, global_step=step)
                    writer_pxl.add_scalar('Pixel loss', pxlLoss, global_step=step)
                    writer_ctx.add_scalar('Context loss', ctxLoss, global_step=step)
                    writer_prc.add_scalar('Perceptual loss', prcLoss, global_step=step)
                    # pick 16 examples
                    img_grid_real = torchvision.utils.make_grid(real[:16])
                    img_grid_fake = torchvision.utils.make_grid(fake[:16])
                    img_grid_raw = torchvision.utils.make_grid(raw[:16])
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake_masked.add_image("Fake Masked", img_grid_fake, global_step=step)
                    writer_fake_raw.add_image("Fake Raw", img_grid_raw, global_step=step)
                step += 1

        # save model at the end of each epoch
        fileName = "epoch_" + str(epoch + 1) + '.pth'
        checkpoint = {'G': gen,
                      'D': disc,
                      'G_state': gen.state_dict(),
                      'D_state': disc.state_dict(),
                      'Optim_G': opt_gen.state_dict(),
                      'Optim_D': opt_disc.state_dict()}
        SaveModel(checkpoint, modelDir, fileName)
        print(f"Models for epoch {epoch + 1} saved")

    # save the final iteration of the model
    final = {'G': gen,
             'D': disc,
             'G_state': gen.state_dict(),
             'D_state': disc.state_dict(),
             'Optim_G': opt_gen.state_dict(),
             'Optim_D': opt_disc.state_dict()}
    # torch.save(final, "v1.pth")


def Generate(file, outputDir, numSamples, maskDir='outputMasks', inputDir='outputSlices'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(file)
    print(checkpoint.keys())
    gen = Generator(Z=Z_dim, imgChannels=Img_channels, features=Features_gen)
    gen.load_state_dict(checkpoint["G_state"])
    disc = Discriminator(imgChannels=Img_channels, features=Features_disc)
    disc.load_state_dict(checkpoint["D_state"])

    # Optimiser Functions
    opt_disc = optim.Adam(params=disc.parameters(),
                          betas=(beta1, beta2),
                          eps=epsilon)

    opt_gen = optim.Adam(params=gen.parameters(),
                         betas=(beta1, beta2),
                         eps=epsilon)
    opt_disc.load_state_dict(checkpoint['Optim_D'])
    opt_gen.load_state_dict(checkpoint['Optim_G'])

    gen.eval().to(device)
    masks = os.listdir(maskDir)
    counter = 0
    noise = torch.rand((Batch_size, Z_dim, 1, 1)).to(device)

    # create folders for fake and real images
    os.mkdir(outputDir)
    os.mkdir(outputDir + '/real')
    os.mkdir(outputDir + '/fake')

    for file in os.listdir(inputDir):
        # select a random mask to apply to the sample
        # idx = random.randrange(len(masks))
        # mask = torch.load(maskDir + '/' + masks[idx]).to(device)
        for m in masks:
            mask = torch.load(maskDir + '/' + m).to(device)
            # load real sample
            real = torch.load(inputDir + '/' + file).to(device)

            with torch.no_grad():
                raw = (torch.abs(gen(x=noise)))
                fake = ((mask * real) + ((1 - mask) * raw * 400))
                # save output to file
                fakePath = outputDir + '/fake/' + str(counter) + '_fake' + m
                realPath = outputDir + '/real/' + str(counter) + '_real.pt'
                torch.save(fake[0], fakePath)
        torch.save(real, realPath)
        counter += 1
        if counter == numSamples:
            break
