import torch
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import tqdm


class Nflow2D:
    def __init__(self, max_iter=400, b_plot=True, grid_size=200, show_iter=100) -> None:
        self.max_iter = max_iter

        self.xx = None
        self.yy = None
        self.zz = None
        self.b_plot = b_plot
        self.grid_size = grid_size
        self.show_iter = show_iter
        # Set up model
        # Define 2D Gaussian base distribution
        base = nf.distributions.base.DiagGaussian(2)

        # Define list of flows
        num_layers = 32
        flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(2, mode='swap'))

        # Construct flow model
        model = nf.NormalizingFlow(base, flows)
        # Move model on GPU if available
        enable_cuda = True
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   and enable_cuda else 'cpu')
        model = model.to(self.device)

        self.model = model

    def fit(self, xs):
        # Get training samples
        x = xs.to(self.device)
        # Train model

        loss_hist = np.array([])

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=5e-4, weight_decay=1e-5)

        for it in tqdm(range(self.max_iter)):
            optimizer.zero_grad()

            # Compute loss
            loss = self.model.forward_kld(x)

            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

            if self.b_plot:
                if self.xx is None:
                    self.xx, self.yy = torch.meshgrid(
                        torch.linspace(-3, 3, self.grid_size), torch.linspace(-3, 3, self.grid_size))
                    zz = torch.cat(
                        [self.xx.unsqueeze(2), self.yy.unsqueeze(2)], 2).view(-1, 2)
                    self.zz = zz.to(self.device)

                # Plot learned distribution
                if (it + 1) % self.show_iter == 0:
                    self.model.eval()
                    log_prob = self.model.log_prob(self.zz)
                    self.model.train()
                    prob = torch.exp(log_prob.to('cpu').view(*self.xx.shape))
                    prob[torch.isnan(prob)] = 0

                    plt.figure(figsize=(15, 15))
                    plt.pcolormesh(self.xx, self.yy,
                                   prob.data.numpy(), cmap='coolwarm')
                    plt.gca().set_aspect('equal', 'box')
                    plt.show()

        if self.b_plot:
            # Plot loss
            plt.figure(figsize=(10, 10))
            plt.plot(loss_hist, label='loss')
            plt.legend()
            plt.show()

        self.model.eval()

    def predict(self, data):
        data = data.to(self.device)
        log_prob = self.model.log_prob(data).to(
            'cpu')  # .view(*self.xx.shape)
        return torch.exp(log_prob)

    def plot(self):
        if self.b_plot:
            # Plot target distribution
            f, ax = plt.subplots(1, 1, sharey=True, figsize=(15, 7))

            # log_prob = self.target.log_prob(self.zz).to('cpu').view(*self.xx.shape)
            # prob = torch.exp(log_prob)
            # prob[torch.isnan(prob)] = 0

            # ax[0].pcolormesh(self.xx, self.yy, prob.data.numpy(), cmap='coolwarm')

            # ax[0].set_aspect('equal', 'box')
            # ax[0].set_axis_off()
            # ax[0].set_title('Target', fontsize=24)

            # Plot learned distribution
            # print(self.zz)
            log_prob = self.model.log_prob(
                self.zz).to('cpu').view(*self.xx.shape)
            # self.model.train()
            prob = torch.exp(log_prob)
            prob[torch.isnan(prob)] = 0

            ax.pcolormesh(self.xx, self.yy,
                          prob.data.numpy(), cmap='coolwarm')

            ax.set_aspect('equal', 'box')
            ax.set_axis_off()
            ax.set_title('Real NVP', fontsize=24)

            plt.subplots_adjust(wspace=0.1)

            plt.show()
        self.model.eval()


if __name__ == '__main__':
    nflow = Nflow1D(max_iter=10, b_plot=False)
    # Define target distribution
    # target = nf.distributions.GaussianMixture(3, 1)
    # target = nf.distributions.UniformGaussian(
    #     2, [1], torch.tensor([1., 2 * np.pi]))
    # Set up target

    class Target:
        def __init__(self, ndim, ind_circ):
            self.ndim = ndim
            self.ind_circ = ind_circ

        def sample(self, n):
            s = torch.randn(n, self.ndim)
            c = torch.rand(n, self.ndim) > 0.6
            s = c * (0.3 * s - 0.5) + (1 - 1. * c) * (s + 1.3)
            u = torch.rand(n, len(self.ind_circ))
            s_ = torch.acos(2 * u - 1)
            c = torch.rand(n, len(self.ind_circ)) > 0.3
            s_[c] = -s_[c]
            s[:, self.ind_circ] = (s_ + 1) % (2 * np.pi) - np.pi
            return s

    # Visualize target
    # target = Target(2, [1])
    # s = target.sample(10000)
    # x = s[:, 0]
    # plt.hist(s[:, 0].data.numpy(), bins=200)
    # plt.show()
    # plt.hist(s[:, 1].data.numpy(), bins=200)
    # plt.show()
    # num_samples = 2 ** 16
    # x = target.sample(num_samples)
    x = torch.tensor(np.array(np.linspace(0, 1, 10).reshape(-1, 1)))
    print(x)
    # exit()
    nflow.fit(x)
    nflow.plot()
    zz = torch.tensor([[1.0], [2.0], [3.0]])
    print(zz)
    print(nflow.predict(zz))

    nflow = Nflow2D(max_iter=10, b_plot=False)
    # Define target distribution
    target = nf.distributions.TwoMoons()
    num_samples = 2 ** 16
    x = target.sample(num_samples)
    nflow.fit(x)
    nflow.plot()
    zz = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    print(zz)
    print(nflow.predict(zz))
