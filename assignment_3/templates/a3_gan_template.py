import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        # Block 1
        self.linear1 = nn.Linear(args.latent_dim, 128)
        
        # Block 2
        self.linear2 = nn.Linear(128, 256)
        self.bnorm2  = nn.BatchNorm1d(256)

        # Block 3
        self.linear3 = nn.Linear(256, 512)
        self.bnorm3  = nn.BatchNorm1d(512)

        # Block 4 
        self.linear4 = nn.Linear(512, 1024)
        self.bnorm4  = nn.BatchNorm1d(1024)

        # Block 5
        self.linear5 = nn.Linear(1024, 784)

        # Activations - ReLU and Sigmoid
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Generate images from z

        # Block 1 
        z = self.linear1(z)
        z = self.relu(z)

        # Block 2 
        z = self.linear2(z) 
        z = self.bnorm2(z)
        z = self.relu(z)

        # Block 3 
        z = self.linear3(z)
        z = self.bnorm3(z)
        z = self.relu(z)

        # Block 4
        z = self.linear4(z)
        z = self.bnorm4(z)
        z = self.relu(z)

        # Block 5 / output
        z = self.linear5(z)
        z = self.tanh(z)

        # Return
        return z


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        # Block 1 
        self.linear1 = nn.Linear(784, 512)

        # Block 2 
        self.linear2 = nn.Linear(512, 256)
        
        # Block 3
        self.linear3 = nn.Linear(256, 1)
        
        # Activations
        self.relu = nn.LeakyReLU(0.2)
        self.sigm = nn.Sigmoid()

    def forward(self, img):
        # return discriminator score for img
        
        # Block 1
        pred = self.linear1(img)
        pred = self.relu(pred)

        # Block 2
        pred = self.linear2(pred)
        pred = self.relu(pred)

        # Block 3
        pred = self.linear3(pred)
        pred = self.sigm(pred)

        # final output \in [0,1]
        return pred

def calc_accuracy(preds, targets):
    print("PREDS: ", preds)
    print("TARGS: ", targets)
    result = torch.sum(preds == targets)/targets.shape[0]
    print(result)
    return result


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):

    # Initialize the losses
    d_loss = torch.nn.BCELoss()
    g_loss = torch.nn.BCELoss()

    # Keep track of losses
    avg_loss_d = 0
    avg_loss_g = 0

    # Main epic loop
    for epoch in range(args.n_epochs):

        for i, (imgs, _) in enumerate(dataloader):

            # Epsilon for numerical stability in some places
            epsilon = 1e-8

            # Set imgs to device and get batch size
            imgs = imgs.to(device)            
            batch_dim = imgs.shape[0]

            # Initialize labels
            true_labels = torch.ones( batch_dim, 1).to(device)
            fake_labels = torch.zeros(batch_dim, 1).to(device)

            # Train Generator
            # ---------------
            
            # Clear the gradient buffers
            optimizer_G.zero_grad()

            # Generate noise and feed it to generator
            noise = torch.randn((imgs.shape[0], args.latent_dim)).to(device)
            gen_imgs = generator.forward(noise)

            # Feed generated images to discriminator
            gen_preds = discriminator.forward(gen_imgs)
            gen_loss = g_loss(gen_preds, true_labels)

            # Propagate the loss backward
            gen_loss.backward()#retain_graph = True)
            optimizer_G.step()

            avg_loss_g += gen_loss.item()

            # Train Discriminator
            # -------------------

            # First we train on real data
            # Clear the gradient buffers
            optimizer_D.zero_grad()

            # Feed real images with some noise to prevent discriminator from converging too quickly
            imgs = imgs.view(-1, 784)

            imgs = imgs + (1/(epoch+1+epsilon)**1.5)*torch.randn_like(imgs)
            real_preds = discriminator.forward(imgs)    

            # Calculate real loss and backward it 
            real_loss = d_loss(real_preds, true_labels)
            real_loss.backward()#retain_graph = True)
            
            # Build disc loss and accuracy
            avg_loss_d += real_loss.item()

            # .. then we train on fake data
            # Feed fake images
            fake_preds = discriminator.forward(gen_imgs.detach())

            # Calculate fake loss and backward it
            fake_loss = d_loss(fake_preds, fake_labels)
            fake_loss.backward()

            # Now we can take optimizer step
            optimizer_D.step()


            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # Print state
                avg_loss_d /= args.save_interval
                avg_loss_g /= args.save_interval
                print(
                    f"[Epoch {epoch}] | D_loss: {avg_loss_d} | G_loss: {avg_loss_g}")
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                #save_imgs = gen_imgs.view(imgs.shape[0], 28, 28)
                save_image(gen_imgs[:25].view(-1,1,28,28),
                           'images/6/{}.png'.format(batches_done),
                            nrow=5, normalize=True)

                # Reset average loss for batches_done
                avg_loss_d = 0
                avg_loss_g = 0
                


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
