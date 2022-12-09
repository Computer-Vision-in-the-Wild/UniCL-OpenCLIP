import torch
import time
from tqdm import tqdm
from datasets import class_map, template_map
from .tokenizer import tokenize


@torch.no_grad()
def extract_text_features(task_list, args=None, model=None, return_numpy=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    zeroshot_weights_list = []

    for task in task_list:
        class_names = class_map.get(task)
        templates = template_map.get(task, ['a photo of a {}'])
        start = time.time()
        model.to(device)
        model.eval()

        zeroshot_weights = []
        for classname in tqdm(class_names, f'Extracting text features with model {"VIT-B32"}.'):  # TODO: need to revise model name
            if type(classname) == list: classname = classname[0]

            texts = [template.format(classname) for template in templates ]   
            texts = tokenize(texts).to(device)
            if args.distributed and not args.horovod:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
         
        if return_numpy:  # default is False
            zeroshot_weights =  zeroshot_weights.cpu().detach().numpy()

        zeroshot_weights_list.append(zeroshot_weights) 

    return zeroshot_weights_list