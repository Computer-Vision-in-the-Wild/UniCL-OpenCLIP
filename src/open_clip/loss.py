import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def targetM(y):
    cap_m = (y == 0).sum()
    cls_m = y[y>0].max()
    y[y==0] = torch.arange(0, cap_m) + cls_m + 1
    return y.view(-1, 1) == y.view(1, -1)

def SoftCE(s, t):
    s = torch.softmax(s, dim=-1)
    loss = - (t * s.log()).sum(dim = -1)
    return (loss/t.sum(dim=-1)).mean()

def change_onehot_to_idx_unicl(labels):

    # convert one hot to index
    one_index = (labels == 1.).nonzero(as_tuple=False).numpy()
    cur_row = one_index[0][0]
    label_list = []
    cur_row = one_index[0][0]
    cur_label = []
    max_length = 0
    for item in one_index:
        if item[0] == cur_row:
            cur_label.append(item[1])
        else:
            label_list.append(cur_label)
            if max_length < len(cur_label):
                max_length = len(cur_label)
            cur_row = item[0]
            cur_label = [item[1]]
    label_list.append(cur_label)
    if max_length < len(cur_label):
        max_length = len(cur_label)

    label_list_index = []
    for label in label_list:
        label_num = 0
        for i in range(len(label)):
            label_num += label[i] * ((0.01) ** i)
        label_list_index.append(round(label_num, 4))

    label_list_index = torch.tensor(label_list_index)
    return label_list_index.view(-1,1) == label_list_index.view(1,-1)


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        labels=None
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        assert labels is None, 'Right now, horovod is not supported for UniCL, we are working on this.'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            if labels is not None:
                all_labels = torch.cat(torch.distributed.nn.all_gather(labels), dim=0)
        else:
            assert labels is None, 'Right now, gather_without_grad is not supported for UniCL, we are working on this.'
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    if labels is None:
        return all_image_features, all_text_features
    else:
        return all_image_features, all_text_features, all_labels


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss


class UniCLLoss(nn.Module):

    def __init__(
            self, local_loss=False,
            gather_with_grad=False, 
            cache_labels=False,
            rank=0, 
            world_size=1, 
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, labels):
        device = image_features.device

        if self.world_size > 1:
            all_image_features, all_text_features, labels = gather_features(
                                    image_features, text_features,
                                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod, labels=labels)
            if self.local_loss:
                logits = logit_scale * all_image_features @ all_text_features.T  # logit_scale: learned temperature parameter
            else:
                raise RuntimeError(f"Right now, local_loss == False is not supported for UniCL, we are working on this!")
        else:
            logits = logit_scale * image_features @ text_features.T  

        # for multi-label dataset such as VOC2007, we need to reformat the labels into one-hot format
        if len(labels.size()) > 1:
            target = change_onehot_to_idx_unicl(labels.cpu()).to(device)
        else:
            target = targetM(labels.cpu()).to(device)

        i2t = SoftCE(logits, target)
        t2i = SoftCE(logits.T, target.T)
        total_loss = (i2t + t2i) / 2
        total_loss = total_loss / self.world_size
        
        return total_loss

