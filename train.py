'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

In our example we will be using data that can be downloaded at:
https://www.kaggle.com/tongpython/cat-and-dog

In our setup, it expects:
- a data/ folder
- train/ and validation/ subfolders inside data/
- cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-X in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 0-X in data/train/dogs
- put the dog pictures index 1000-1400 in data/validation/dogs

We have X training examples for each class, and 400 validation examples
for each class. In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```

NOTE: This code uses PyTorch for training the model.
      You can find the Keras implementation in this repo:
         https://github.com/iterative/example-versioning.git
'''

import numpy as np
import sys
import os

import torch
import torch.optim as optim

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, models, transforms

torch.manual_seed(1)

pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'model.pth'
train_data_dir = os.path.join('data', 'train')
validation_data_dir = os.path.join('data', 'validation')
cats_train_path = os.path.join(path, train_data_dir, 'cats')
nb_train_samples = 2 * len([name for name in os.listdir(cats_train_path)
                            if os.path.isfile(
                                os.path.join(cats_train_path, name))])
nb_validation_samples = 800
epochs = 30
batch_size = 10
dog_label = 0
cat_label = 1

def get_vgg16_model_pretrained_exclude_top():
    model = models.vgg16(pretrained=True)
    return model.features[:]

def load_dataset_from_data_dir():
    data_transforms = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(), # Rescales data by 1./255
    ])
    train_data = datasets.ImageFolder(train_data_dir, transform=data_transforms)
    validation_data = datasets.ImageFolder(validation_data_dir, transform=data_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader

def load_dataset_from_numpy_data(data_x, data_y, shuffle=False):
    tensor_x = torch.Tensor(data_x) # transform to torch tensor
    tensor_y = torch.Tensor(data_y)

    out_dataset = TensorDataset(tensor_x,tensor_y)
    return DataLoader(out_dataset, batch_size=batch_size, shuffle=shuffle)

def model_predict_generator(model, dataloader):
    model.train()
    features = None
    for ii, (data, labels) in enumerate(dataloader):
        # Permute as output from this model is of dimension [BCWH], permute it to [BWHC]
        out = model(data).permute(0, 2, 3, 1)
        if features is None:
            features = out
        else:
            features = torch.cat((features, out))
    return features.detach().numpy()

def model_fit(model, train_loader, validation_loader, epochs, batch_size, optimizer, criterion, metrics_file):
    metrics_f = open(metrics_file, "w")
    metrics_f.write(f'epoch,accuracy,loss,val_accuracy,val_loss,dog_accuracy,cat_accuracy\n')
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        pbar = tqdm(train_loader)

        model.train()
        train_correct, train_loss = 0.0, 0.0
        for batch_idx, (train_data, train_label) in enumerate(pbar):
            optimizer.zero_grad()
            train_pred = model(train_data).squeeze()
            train_target = train_label.squeeze()
            loss = criterion(train_pred, train_target)
            loss.backward()
            optimizer.step()
            train_loss += loss
            train_correct += (train_pred == train_target).sum()
            pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * (train_correct / len(train_loader.dataset))

        with torch.no_grad():
            model.eval()
            val_correct, val_loss = 0.0, 0.0
            dog_total, cat_total = 0.0, 0.0
            dog_val_correct, cat_val_correct = 0.0, 0.0
            for batch_idx, (val_data, val_label) in enumerate(validation_loader):
                val_pred = model(val_data).squeeze()
                val_target = val_label.squeeze()
                l = criterion(val_pred, val_target).item()
                val_loss += l
                val_correct += (val_pred == val_target).sum()
                for idx, val_l in enumerate(val_target):
                    if val_l == dog_label:
                        if val_pred[idx] == val_l:
                            dog_val_correct += 1
                        dog_total += 1
                    else:
                        if val_pred[idx] == val_l:
                            cat_val_correct += 1
                        cat_total += 1
            val_loss /= len(validation_loader.dataset)
            val_accuracy = 100. * (val_correct / len(validation_loader.dataset))
            dog_val_accuracy = 100. * (dog_val_correct / dog_total)
            cat_val_accuracy = 100. * (cat_val_correct / cat_total)
            print('\nVal_Loss: {:.4f}, Val_Accuracy: {:.0f}%, Dog_Val_Accuracy: {:.0f}%, Cat_Val_Accuracy: {:.0f}%\n'.format(
                val_loss, val_accuracy, dog_val_accuracy, cat_val_accuracy
            )) 
            metrics_f.write(f'{epoch},{train_accuracy},{train_loss},{val_accuracy},{val_loss},{dog_val_accuracy},{cat_val_accuracy}\n')


def save_bottlebeck_features():
    model = get_vgg16_model_pretrained_exclude_top()

    train_loader, validation_loader = load_dataset_from_data_dir()

    bottleneck_features_train = model_predict_generator(model, train_loader)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)
    print("Saved bottleneck train features")

    bottleneck_features_validation = model_predict_generator(model, validation_loader)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    print("Saved bottleneck validation features")

class CatDogNN(nn.Module):
    def __init__(self):
        super(CatDogNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4*4*512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'), allow_pickle=True)
    train_labels = np.array(
        [0] * (int(nb_train_samples / 2)) + [1] * (int(nb_train_samples / 2)))
    train_loader = load_dataset_from_numpy_data(train_data, train_labels, shuffle=True)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'), allow_pickle=True)
    validation_labels = np.array(
        [0] * (int(nb_validation_samples / 2)) +
        [1] * (int(nb_validation_samples / 2)))
    validation_loader = load_dataset_from_numpy_data(validation_data, validation_labels, shuffle=False)

    model = CatDogNN()

    model_fit(
            model,
            train_loader,
            validation_loader,
            epochs,
            batch_size,
            optim.RMSprop(model.parameters()),
            nn.BCELoss(),
            "metrics.csv",
    )

    torch.save(model.state_dict(), top_model_weights_path)


save_bottlebeck_features()
train_top_model()
