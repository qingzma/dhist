import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta


class Nflow1D:
    def __init__(self, b_plot=True, enable_cuda=True) -> None:
        # self.train_loader = None
        # self.test_loader = None
        self.pdf = None
        self.b_plot = b_plot
        self.min = None
        self.max = None
        self.target_distribution = None

        device = 'cpu'
        if enable_cuda:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
        self.device = torch.device(device)
        self.target_distribution = Uniform(torch.tensor(0.0).to(
            self.device), torch.tensor(1.0).to(self.device))

    def fit(self, x):
        x = x.astype('float32')
        self.min = np.min(x)
        self.max = np.max(x)
        train_loader = data.DataLoader(NumpyDataset(
            x, self.device), batch_size=128, shuffle=True)

        self.pdf, train_losses = train_and_eval(
            50, 5e-3, train_loader, self.target_distribution, self.device)
        self.pdf.eval()

        if self.b_plot:
            plt.plot(train_losses, label='train_loss')
            plt.legend()
            plt.show()

    def predict(self, x):
        z, dz_by_dx = self.pdf(torch.FloatTensor(x).to(self.device))
        px = torch.exp(self.target_distribution.log_prob(z) +
                       dz_by_dx.log()).detach().to('cpu').numpy()
        return px

    def plot(self):
        x = np.linspace(self.min, self.max, 100)
        px = self.predict(x)
        # print(x)
        # print(px)
        # exit()
        plt.plot(x, px)
        plt.title('Learned probability distribution')
        plt.show()


class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(
            torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(
            torch.ones(n_components), requires_grad=True)

    def forward(self, x):
        x = x.view(-1, 1)
        weights = self.weight_logits.softmax(dim=0).view(1, -1)
        distribution = Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1)
        return z, dz_by_dx


def loss_function(target_distribution, z, dz_by_dx):
    log_likelihood = target_distribution.log_prob(z) + dz_by_dx.log()
    return -log_likelihood.mean()


def train(model, train_loader, optimizer, target_distribution):
    model.train()
    for x in train_loader:
        z, dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    for x in data_loader:
        z, dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, dz_by_dx)
        total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()


def train_and_eval(epochs, lr, train_loader, target_distribution, device='cpu'):
    flow = Flow1d(n_components=5)
    flow = flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    train_losses = []
    for epoch in range(epochs):
        train(flow, train_loader, optimizer,
              target_distribution)
        train_losses.append(eval_loss(flow, train_loader, target_distribution))
        # test_losses.append(eval_loss(flow, test_loader, target_distribution))
    return flow, train_losses


def generate_mixture_of_gaussians(num_of_points):
    n = num_of_points // 2
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(num_of_points-n,))
    return np.concatenate([gaussian1, gaussian2])


class NumpyDataset(data.Dataset):
    def __init__(self, array, device='cpu'):
        super().__init__()
        self.array = torch.FloatTensor(array).to(device)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]


if __name__ == '__main__':
    # device = 'cpu'
    # enable_cuda = True
    # if enable_cuda:
    #     if torch.cuda.is_available():
    #         device = 'cuda'
    #     elif torch.backends.mps.is_available():
    #         device = 'mps'
    # device = torch.device(device)

    n_train, n_test = 20000, 1000
    train_data = generate_mixture_of_gaussians(n_train)
    # print(train_data)
    # print(type(train_data))
    # test_data = generate_mixture_of_gaussians(n_test)

    # train_loader = data.DataLoader(NumpyDataset(
    #     train_data, device), batch_size=128, shuffle=True)
    # test_loader = data.DataLoader(NumpyDataset(
    #     test_data), batch_size=128, shuffle=True)

    # _, axes = plt.subplots(1, 2, figsize=(12, 4))
    # _ = axes[0].hist(train_loader.dataset.array, bins=50)
    # # _ = axes[1].hist(test_loader.dataset.array, bins=50)
    # _ = axes[0].set_title('Training data')
    # _ = axes[1].set_title('Testing data')
    # plt.show()

    flow = Nflow1D(enable_cuda=False)
    flow.fit(train_data)
    flow.plot()

    # target_distribution = Uniform(0.0, 1.0)
    # flow, train_losses = train_and_eval(
    #     50, 5e-3, train_loader, target_distribution)

    # _ = plt.plot(train_losses, label='train_loss')
    # # _ = plt.plot(test_losses, label='test_loss')
    # plt.legend()
    # plt.show()

    # x = np.linspace(-3, 3, 1000)
    # with torch.no_grad():
    #     z, dz_by_dx = flow(torch.FloatTensor(x))
    #     px = (target_distribution.log_prob(z) +
    #           dz_by_dx.log()).exp().cpu().numpy()

    # _, axes = plt.subplots(1, 2, figsize=(12, 4))
    # _ = axes[0].grid(), axes[1].grid()
    # _ = axes[0].plot(x, px)
    # _ = axes[0].set_title('Learned probability distribution')

    # _ = axes[1].plot(x, z)
    # _ = axes[1].set_title('x -> z')
    # plt.show()
