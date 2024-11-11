import torch
import math
import torch.nn.functional as F
import torch.distributions as D
import torch.nn as nn

class StoLayer(nn.Module):
    def sto_init(self, 
                 n_components=2,
                 prior_mean=1.0, 
                 prior_std=0.40, 
                 post_mean_init=(1.0, 0.05), 
                 post_std_init=(0.25, 0.10),
                 ):
        self.prior_mean=nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std=nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        self.n_components = n_components
        self.post_mean_init = post_mean_init
        self.post_std_init = post_std_init

        latent_shape = [1] * (self.weight.ndim-1)
        latent_shape[0] = sum(self.weight.shape[:2])   

        self.post_mean = nn.Parameter(torch.ones(n_components, *latent_shape), requires_grad=True)
        self.post_std = nn.Parameter(torch.ones(n_components, *latent_shape), requires_grad=True)

        nn.init.normal_(self.post_mean, post_mean_init[0], post_mean_init[1])
        nn.init.normal_(self.post_std, post_std_init[0], post_std_init[1])
    
    def _entropy_lower_bound(self, mean, std):
        """
        calculate entropy lower bound for mixed Gaussian
        mean, std: [n_components, *shape_of_feature]
        formula reference: https://www.mdpi.com/1099-4300/19/7/361
        """
        cond_entropy = D.Normal(mean, std).entropy().mean(dim=0).sum()

        # pairwise_mean_dist = [component_index1, component_index2, size_of_feature]
        mean_flat = mean.flatten(1)
        std_flat = std.flatten(1)
        var_flat = std_flat.square()
        logstd = std_flat.log().sum(1)

        pairwise_mean_dist = mean_flat.unsqueeze(1) - mean_flat.unsqueeze(0)
        pairwise_var_sum = var_flat.unsqueeze(1) + var_flat.unsqueeze(0)
        pairwise_std_logprod = logstd.unsqueeze(1) + logstd.unsqueeze(0)

        c_a_diverge = (pairwise_mean_dist.square()/pairwise_var_sum).sum(2)/4 +\
                        0.5*(torch.log(0.5*pairwise_var_sum).sum(2)-pairwise_std_logprod)
        second_part = torch.logsumexp(-c_a_diverge, dim=1) - torch.log(torch.tensor(mean.size(0), dtype=mean.dtype, device=mean.device))

        entropy_lower_bound = cond_entropy - second_part.mean(0)
        return entropy_lower_bound

    def _kl(self, kl_scale=None):
        """
        estimate kl divergence between mixed Gaussian and Gaussian
        """
        prior = D.Normal(self.prior_mean, self.prior_std)
        post_mean = self.post_mean
        post_std = F.softplus(self.post_std)
        post_std = torch.clamp(post_std, min=1e-6)
        post = D.Normal(post_mean, post_std)
        cross_entropy = (D.kl_divergence(post, prior) + post.entropy()).flatten(1).sum(1).mean()
        kl = cross_entropy - self._entropy_lower_bound(post_mean, post_std)
        if kl_scale is None:
            # kl_scale = 1/(math.sqrt(post_mean.numel() + 1e-10))
            kl_scale = 1
        kl = kl * kl_scale
        return kl

    def _entropy(self, entropy_scale=None):
        """
        estimate posterior entropy with its upper bound
        """
        mean = self.post_mean
        std = F.softplus(self.post_std)
        std = torch.clamp(std, min=1e-6)
        entropy = self._entropy_lower_bound(mean, std)
        if entropy_scale is None:
            entropy_scale = 1/(math.sqrt(mean.numel() + 1e-10))
        entropy = entropy * entropy_scale
        return entropy

    def sto_extra_repr(self):
        info=(f"n_components={self.n_components}, " 
              f"prior_mean={self.prior_mean.detach().item()}, prior_std={self.prior_std.detach().item()}, "
              f"posterior_mean_init={self.post_mean_init}, posterior_std_init={self.post_std_init}, " )
        return info

class StoSequential(nn.Sequential, StoLayer):
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, input, gs_indices):
        for module in self:
            if isinstance(module, StoLayer):
                input = module(input, gs_indices)
            else:
                input = module(input)
        return input

class StoLinear(nn.Linear, StoLayer):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool=False,
                 n_components=2, 
                 prior_mean=1.0, 
                 prior_std=0.30, 
                 post_mean_init=(1.00, 0.05), 
                 post_std_init=(0.40, 0.02), 
                 ):
        super().__init__(in_features, out_features, bias)
        self.sto_init(n_components, prior_mean, prior_std, post_mean_init, post_std_init)

    def forward(self, x):
        gs_indices = torch.arange(x.size(0), dtype=torch.long, device=x.device) % self.n_components
        self.post_std.data = torch.clamp(self.post_std.data, min=-10.0)
        post_std = F.softplus(self.post_std)
        post_std = torch.clamp(post_std, min=1e-6)
        epsilon = torch.randn((x.size(0), *self.post_mean.shape[1:]), device=x.device, dtype=x.dtype)
        random_weight = self.post_mean[gs_indices] + post_std[gs_indices] * epsilon
        x = x * random_weight[:, :x.shape[1]]
        x = super().forward(x)
        x = x * random_weight[:, -x.shape[1]:]
        return x
    
    def extra_repr(self):
        return f'{super().extra_repr()}, {self.sto_extra_repr()}'