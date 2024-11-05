import os
import warnings
from collections import abc
import numpy as np
import torch
import torch.nn.functional as F
from importlib import import_module
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    """
    Args: 
        output: output.shape=N*L, N=batch_size, L=num_of_points (Note: for shape classification, output.shape = N)
        target: target.shape=output.shape
        K: num_of_classes
        ignore_index: class to ignore(e.g. background), regard as true prediction

    Return:
        area_intersection: number of intersection of each_class, len(area_interaction)=K
    """
    # 'K' classes, output and target sizes are N or N * L, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def get_linear_weight(current_epoch, max_epoch, weight_init=1e-2, weight_final=1.0):
    epoch_weight = weight_init + (current_epoch/max_epoch)*(weight_final-weight_init)
    return epoch_weight

def point_wise_entropy(logits, type='predictive'):
    # prob.shape: [N, n_samples, C]
    prob = F.softmax(logits, dim=-1)
    if type=='predictive':
        prob = torch.mean(prob, dim=1)
        entropy = - torch.sum(torch.mul(prob, torch.log(prob + 1e-10)), dim=-1)
    elif type=='aleatoric':
        prob = prob.transpose(0,1)
        entropy = - torch.sum(torch.mul(prob, torch.log(prob + 1e-10)), dim=-1)
        entropy = torch.mean(entropy, dim=0)
    return entropy

class ECE(torch.nn.Module):
    """
    expected calibration error, measure how reliable confidence score is.
    """
    def __init__(self, n_bins=15):
        super().__init__()
        bins = torch.linspace(0, 1, n_bins+1)
        self.bin_lowers = bins[:-1]
        self.bin_uppers = bins[1:]
    
    def forward(self, preds, labels, plot=False):
        # preds.shape = [len(testset), n_classes], labels.shape[len(testset)], 
        # note: preds need to be normalized to [0,1]
        confidences, preds = torch.max(preds, dim=1)
        correct_preds = preds.eq(labels)

        ece = 0.
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            sample_in_bin = confidences.gt(bin_lower.item())*confidences.le(bin_upper.item())
            prop_in_bin = sample_in_bin.float().mean()
            if  prop_in_bin.item() > 0:
                acc_in_bin = correct_preds[sample_in_bin].float().mean()
                avg_confidence_in_bin = confidences[sample_in_bin].mean()
                diff_in_bin = torch.abs(avg_confidence_in_bin - acc_in_bin) * prop_in_bin
                ece += diff_in_bin.item()
        return ece
    
    def plot_calibration(self, acc_per_bin, conf_per_bin):
        # bin_centers = (self.bin_lowers + self.bin_uppers) / 2
        # plt.figure(figsize=(10, 6))
        
        # bar_width = 0.4
        # # Plot the accuracy
        # plt.bar(bin_centers.numpy() - bar_width/2, acc_per_bin, width=bar_width, label='Accuracy', alpha=0.6, color='blue')
        # # Plot the confidence
        # plt.bar(bin_centers.numpy() + bar_width/2, conf_per_bin, width=bar_width, label='Mean Confidence', alpha=0.6, color='red')

        # # Set labels and title
        # plt.xlabel("Confidence")
        # # plt.ylabel("Accuracy")
        # plt.title("Calibration Plot (Accuracy vs. Confidence)")
        # plt.legend()
        # plt.text(0.95, 0.05, f"ECE: {ece_score:.4f}", 
        #      verticalalignment='bottom', horizontalalignment='right', 
        #      transform=plt.gca().transAxes, 
        #      color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
        # plt.savefig(save_path)
        # plt.close()

def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items. e.g. int64, list
        seq_type (type, optional): Expected sequence type. e.g. list, tuple

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(['os.path', 'sys'])
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported