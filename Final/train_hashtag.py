import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torch.utils.data as Data
#from build_vocab import Vocabulary
from model import EncoderCNN, EncoderRNN, Model
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import linecache
import cv2
from torch.autograd import Variable

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
f = open('./data/new_file_tag.txt','r')
log_file = open('log.txt','w+')
line_ls = f.readlines()
class traingclass(Data.Dataset):
    """docstring for ClassName"""
    def __init__(self):
        pass
    def __len__(self):
        return 46461
    def __getitem__(self,index):
        #print(index)
        line_opt = line_ls[index]
        #print(line_opt)
        line_opt = line_opt.split(',')
        #print(line_opt)
        images_fil = line_opt[0]
        tag = line_opt[1]
        tag = tag.split(' ')
        tag = np.array(tag)
        #print(tag)
        tag2 = np.zeros((20,))
        for x in range(len(tag)):
            tag2[x] = tag[x]
        #print(tag2)
        im = cv2.imread(images_fil)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
        im = im.transpose(2,0,1)       
        return (im,tag2)

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Build data loader
    # data_loader = get_loader(args.image_dir, args.instance_path,
    #                          transform, args.batch_size,
    #                          shuffle=True, num_workers=args.num_workers) 

    #all_object_ids = data_loader.dataset.inverse_object_id_mapping.keys()
    num_objects = 203
    # with open(args.inverse_object_id_mapping, "wb") as f:
    #     pickle.dump(data_loader.dataset.inverse_object_id_mapping, f)
    train_x = traingclass()
    data_loader = torch.utils.data.DataLoader(train_x,batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    # Build the models
    
    encoderCNN = EncoderCNN(args.embed_size).to(device)
    encoderRNN = EncoderRNN(num_objects, args.embed_size, args.hidden_size).to(device)
    model = Model(num_objects, args.embed_size).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    params = list(encoderRNN.parameters()) + list(encoderCNN.linear.parameters()) + list(encoderCNN.bn.parameters()) + list(model.parameters())
    optimizer = torch.optim.RMSprop(params, lr=args.learning_rate, weight_decay=0.0001, momentum=0.91)
    
    teacher_forcing_ratio = 0.5
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, objects) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images.float()
            objects = Variable(objects).long()
            #objects = objects.float()
            images = images.to(device)
            targets = objects.to(device)
            #print(targets.shape)
            #targets = pack_padded_sequence(objects, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            image_features = encoderCNN(images)
            h0 = torch.zeros((1, args.batch_size, args.hidden_size)).to(device)
            c0 = torch.zeros((1, args.batch_size, args.hidden_size)).to(device)
            loss = 0
            
            teacher_forcing = True if np.random.random() > teacher_forcing_ratio else False

            input = targets[:,0].unsqueeze(1)
            #print(input.size())
            if teacher_forcing:
                for j in range(20-1):
                    hashtag_features, (h0, c0), Ul = encoderRNN(input , h0, c0)
                    outputs = model(image_features, hashtag_features, Ul)
                    loss += criterion(outputs, targets[:, j+1])
                    input = targets[:, j+1].unsqueeze(1)
            else:
                for j in range(20-1):
                    hashtag_features, (h0, c0), Ul = encoderRNN(input , h0, c0)
                    outputs = model(image_features, hashtag_features, Ul)
                    loss += criterion(outputs, targets[:, j+1])
                    _, top1 = outputs.topk(1, dim=1)
                    input = top1

            encoderCNN.zero_grad()
            encoderRNN.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())),file=log_file) 
            # Save the model checkpoints
            if (i+1) % args.save_step == 0 and (epoch+1) % 3 == 0:
                torch.save(encoderCNN.state_dict(), os.path.join(
                    args.model_path, 'encoderCNN-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoderRNN.state_dict(), os.path.join(
                    args.model_path, 'encoderRNN-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'model-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/tmp2/victai/cognitive_computing/CnnRnn/chengyu_models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--image_dir', type=str, default='../data/resized2014', help='directory for resized images')
    parser.add_argument('--instance_path', type=str, default='../data/annotations/instances_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1400, help='step size for saving trained models')
    parser.add_argument('--inverse_object_id_mapping', type=str, default='./inverse_object_id_mapping.pkl')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=64, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=3, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    args = parser.parse_args()
    print(args)
    main(args)
