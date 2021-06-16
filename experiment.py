#!/usr/bin/env python3

import os
from tempfile import gettempdir
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch

# data
device = "cuda:0"
dim = 28 ** 2
tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.15], std=[0.3]),
])
prefix = os.path.join(gettempdir(), "mnist_")
train_set = datasets.MNIST(prefix+"training", download=True, train=True, transform=tr)
test_set = datasets.MNIST(prefix+"testing", download=True, train=False, transform=tr)

# architecture
def new_digit_classifier():
    return nn.Sequential(
        nn.Linear(dim, dim // 8),
        nn.LeakyReLU(0.1),
        nn.BatchNorm1d(dim // 8),
        nn.Linear(dim // 8, dim // 16),
        nn.LeakyReLU(0.1),
        nn.BatchNorm1d(dim // 16),
        nn.Linear(dim // 16, 10, bias=False)
    )

# create a few tasks
tasks = [np.random.permutation(dim) for _ in range(10)]

# train a classifier for each task
progress_bar = tqdm(enumerate(tasks), total=len(tasks))
loss_func = nn.CrossEntropyLoss()
classifiers = []
for task_i, task in progress_bar:
    running_loss = 1.0
    classifier = new_digit_classifier()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    # training loop
    for epoch in range(3):
        loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
        for images, labels in loader:
            features = images.view(images.size(0), dim)[:, task]
            y_pred = classifier(features)
            loss = loss_func(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()
            progress_bar.set_description("loss: %.4f" % running_loss)

    # keep classifier for evaluation
    classifiers.append(classifier.eval().to(device))

# evaluation
confidence_hits = 0
task_conditioned = 0
total = 0
with torch.no_grad():
    loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False)
    for images, labels in loader:
        labels = labels.to(device)
        features = images.view(images.size(0), dim).to(device)
        for task_i, task in tqdm(enumerate(tasks), total=len(tasks)):
            swapped = features[:, task]
            y_pred = torch.cat([
                model(swapped).unsqueeze(2) for model in classifiers
            ], dim=2)
            proba = torch.softmax(y_pred, dim=1)
            # confidence method
            highest_confidence, _ = proba.max(dim=1)
            _, confident_model = highest_confidence.max(dim=1)
            confidence_hits += (confident_model == task_i).float().sum().item()
            # label-conditioned method
            label_proba = proba[np.arange(labels.size(0)), labels, :]
            _, best_match = label_proba.max(dim=1)
            task_conditioned += (best_match == task_i).float().sum().item()
            total += len(labels)

confidence_hits /= total
task_conditioned /= total

print("confidence", confidence_hits)
print("my method", task_conditioned)
