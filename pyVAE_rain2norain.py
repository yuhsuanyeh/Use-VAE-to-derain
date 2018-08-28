#--------------------------------------------------Hi Andy! This code is for VAE rain to norain ,I believe you can understand what the code mean ----------------------------
import torch
import random
import cv2
import torch.nn as nn
import torch.utils.data as Data
from torchvision.utils import save_image
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
# torch.manual_seed(1)    # reproducible
def progbar(curr, total, full_progbar, is_done) :
    """
        Plot progress bar on terminal
        Args :
            curr (int) : current progress
            total (int) : total progress
            full_progbar (int) : length of progress bar
            is_done (bool) : is already done
    """
    frac = curr/total
    filled_progbar = round(frac*full_progbar)

    if is_done == True :
        print('\r|'+'#'*full_progbar + '|  [{:>7.2%}]'.format(1) , end='')
    else :
        print('\r|'+'#'*filled_progbar + '-'*(full_progbar-filled_progbar) + '|  [{:>7.2%}]'.format(frac) , end='')

np.set_printoptions(threshold=np.nan)

# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 4
NUM_SHOW_IMG = 4
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
STEP_NUM = 24000

#Star to load data
train_data = torchvision.datasets.ImageFolder(
    './project_derain/train_rain/',
    transform=transforms.Compose([
        #transforms.Resize((256, 256), 3),
        transforms.ToTensor()
    ])
)

train_data_GT = torchvision.datasets.ImageFolder(
    './project_derain/train_norain/',
    transform=transforms.Compose([
        #transforms.Resize((256, 256), 3),
        transforms.ToTensor()
    ])
)

print('train_data is: ',train_data)
#print('train_data[:BATCH_SIZE] is: ',train_data[:BATCH_SIZE])
#print('train_data[:BATCH_SIZE] size is: ',train_data[:BATCH_SIZE].size())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
print('train_loader is: ',train_loader) #train_ltoader is:  <torch.utils.data.dataloader.DataLoader object at 0x0000028E0D8A3BA8>
print('train_loader type is: ',type(train_loader))

train_loader_GT = Data.DataLoader(dataset=train_data_GT, batch_size=BATCH_SIZE, shuffle=False)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(         # input shape (3, 512, 512)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=64,            # n_filters
                kernel_size=3,              # filter size
                stride=2,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 512, 512)
            nn.LeakyReLU(),                      # activation
        
            nn.Conv2d(64, 128, 3, 2, 1),     # output shape (32, 256, 256)
            nn.BatchNorm2d(num_features = 128, affine = True),
            nn.LeakyReLU(),                      # activation
    
            nn.Conv2d(128, 256, 3, 2, 1),     # output shape (64, 128, 128)
            nn.BatchNorm2d(num_features = 256, affine = True),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, 3, 2, 1),     # output shape (64, 128, 128)
            nn.BatchNorm2d(num_features = 512, affine = True),
            nn.LeakyReLU(),

            nn.Conv2d(512, 1024, 3, 2, 1),     # output shape (64, 128, 128)
            nn.BatchNorm2d(num_features = 1024, affine = True),
            nn.LeakyReLU(),
            )                      
            
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                ),
            nn.BatchNorm2d(num_features = 512, affine = True),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(num_features = 256, affine = True),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(num_features = 128, affine = True),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(num_features = 64, affine = True),
            nn.LeakyReLU(),
        
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.LeakyReLU(),
            )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

VAE = VAE().cuda()
optimizer = torch.optim.Adam(VAE.parameters(), lr=LR)
loss_func = nn.L1Loss().cuda()

# initialize figure
f, a = plt.subplots(2, NUM_SHOW_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

img_ground_list = None
from scipy import misc
'''
# test
img_rain_list = []
img_nonrain_list = []
for i, data in enumerate(train_loader,0):
    images, labels = data
    for idx in range(BATCH_SIZE) :
        if labels[idx] == 0 :
            img_nonrain_list.append(images[idx])
        else : 
            img_rain_list.append(images[idx])

    print('img_rain_list 0 is: {}'.format(img_nonrain_list[0]))
    print('img_rain_list 0 size is: {}'.format(img_nonrain_list[0].shape))
    img_nonrain_list[0] = img_nonrain_list[0].transpose(0,1).transpose(1,2) #chahge (3,512,512) to (512,512,3)
    img_nonrain_list[0].data.cpu().numpy() #torch to array(not necessary)
    print('img_rain_list[0] shape: {}'.format(img_nonrain_list[0].shape)) 
    print('img_rain_list len is: {}'.format(len(img_rain_list)))
    print('img_nonrain_list len is: {}'.format(len(img_nonrain_list)))
#    misc.imsave('YOHEYNO.jpg', img_rain_list[0]) #save array format image
    input()
'''
# original data (first row) for viewing
for i, data in enumerate(train_loader,0):
    images, labels = data
    img_ground_list = images[:NUM_SHOW_IMG] #0-1 float
#    print(A.size()) #torch.Size([5, 3, 512, 512])
    break 

for i in range(NUM_SHOW_IMG):
    a[0][i].imshow(transforms.ToPILImage()(img_ground_list[i]))
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

random_index = list(range(12000))
random.shuffle(random_index)

for epoch in range(EPOCH):
    progress = 0  
    train_loss = 0    
    for step, (b_img, b_label) in enumerate(train_loader):
        (c_img, c_label) = iter(train_loader_GT).next()
        # print('step: ',step)
        # print('img ',images)
        # print('img size is: {}'.format(images.size())) #torch.Size([16, 3, 512, 512])
        # print('img 0 size is: {}'.format(images[0].size())) #torch.Size([3, 512, 512])
        # print('label',labels) # tensor([ 0,  0,  1,  1,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0])
        # print('label size is: {}'.format(labels.size())) #torch.Size([16])
        # b0 = b_img[0].transpose(0,1).transpose(1,2).data.cpu().numpy()
        # b1 = b_img[1].transpose(0,1).transpose(1,2).data.cpu().numpy()
        # b2 = b_img[2].transpose(0,1).transpose(1,2).data.cpu().numpy()
        # b3 = b_img[3].transpose(0,1).transpose(1,2).data.cpu().numpy()
        # c0 = c_img[0].transpose(0,1).transpose(1,2).data.cpu().numpy()
        # c1 = c_img[1].transpose(0,1).transpose(1,2).data.cpu().numpy()
        # c2 = c_img[2].transpose(0,1).transpose(1,2).data.cpu().numpy()
        # c3 = c_img[3].transpose(0,1).transpose(1,2)

        # misc.imsave('b0.jpg', b0) #save array format image
        # misc.imsave('b1.jpg', b1) #save array format image
        # misc.imsave('b2.jpg', b2)
        # misc.imsave('b3.jpg', b3)
        # misc.imsave('c0.jpg', c0)
        # misc.imsave('c1.jpg', c1)
        # misc.imsave('c2.jpg', c2)
        # misc.imsave('c3.jpg', c3)
    
        b_x = b_img.cuda()
        b_y = c_img.cuda()

                # print('step: ',step)
                # print('b_img ',b_img)
                # print('b_img size is: {}'.format(b_img.size())) #torch.Size([16, 3, 512, 512])
                # print('b_label',b_label) # tensor([ 0,  0,  1,  1,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0])
                # print('b_label size is: {}'.format(b_label.size())) #torch.Size([16])
                # input()
                # b_x = b_img.cuda()#Variable(x.view(-1, 3*512*512))   # batch x, shape (batch, 512*512)
                # b_y = c_img.cuda()
    
        recon_batch, mu, logvar = VAE(b_x)
        loss = loss_function(recon_batch, img, mu, logvar)
                #if step % STEP_NUM == 0:
                   #img_to_save = decoded.data
                   #save_image(img_to_save,'res/%s-%s.jpg'%(epoch,step))
                   #io.imsave('res/{}.jpg'.format(epoch),img_to_save[0])

        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        train_loss += loss.data[0]
        optimizer.step()                    # apply gradients

                    #print('[{}][{}/{}]'.format(epoch, step, STEP_NUM))

        if step % STEP_NUM == 0:
            #print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(b_x),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                loss.data[0] / len(b_x)))
            progbar(progress, STEP_NUM, 40, (progress == STEP_NUM-1))
            progress += 1

            if step % 60 == 0:            
                    # plotting decoded image (second row)
                _, decoded_data = VAE(img_ground_list.cuda())

                for i in range(NUM_SHOW_IMG):
                    '''
                    final_decoded_data = torch.mul(decoded_data.data[i].detach(), 255.0)
                    #final_decoded_data = final_decoded_data.type(torch.ByteTensor).cpu().numpy()
                    #final_decoded_data = transforms.ToPILImage()(final_decoded_data)
                    final_decoded_data = np.reshape(final_decoded_data.type(torch.ByteTensor).cpu().numpy(), (256, 256, 3))
                    #final_decoded_datas = final_decoded_datas.type(torch.ByteTensor).cpu().numpy()
                    final_decoded_data = transforms.ToPILImage(mode = 'RGB')(final_decoded_data)
                    #final_decoded_data = final_decoded_datas[i]
                    '''
                    #To visiual training process , you need to save the result first(in the 'res' folder),and then read the result to visual
                    img_to_save = decoded_data.data[i]
                    save_image(img_to_save, 'VAE_output/{}-{}-{}.jpg'.format(i, epoch, step))
                    img_tmp = cv2.imread('VAE_output/{}-{}-{}.jpg'.format(i, epoch, step))
                    img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
                    a[1][i].clear()
                    a[1][i].imshow(img_tmp)
                    a[1][i].set_xticks(())
                    a[1][i].set_yticks(())

    
                plt.draw()
                plt.pause(0.05)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))
                       


plt.ioff()
plt.show()