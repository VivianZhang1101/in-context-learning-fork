import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, w1=0.9, w2=0.1):
        xs_b = torch.zeros(b_size, n_points, self.n_dims)

        if seeds is not None:
            assert len(seeds) == b_size
            generator = torch.Generator()

        for i in range(b_size):
            if seeds is not None:
                generator.manual_seed(seeds[i])

            for j in range(n_points):
                sample1 = (torch.randn(self.n_dims, generator=generator)) * w1 if seeds else (torch.randn(self.n_dims)) * w1
                sample2 = (torch.randn(self.n_dims, generator=generator)) * w2 if seeds else (torch.randn(self.n_dims)) * w2
                xs_b[i, j] = sample1

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        return xs_b