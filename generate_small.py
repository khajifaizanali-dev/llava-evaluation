
from datetime import datetime
from argparse import ArgumentParser

import json, os
from torch.utils.data import Dataset, DataLoader
import torch

from utils import get_worker_class, MileBenchDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import gc


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data/MileBench')
    parser.add_argument('--dataset_name', default='data/sample.json')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--bsz', default=1, type=int)
    parser.add_argument('--batch-image', default=1, type=int)
    parser.add_argument('--combine_image', default=None, type=int, help='Use combined N images for evaluation.')
    parser.add_argument('--model_configs', default='configs/model_configs.yaml')
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    args.output_pth = os.path.join(args.output_dir, f"{args.model_name}/{args.dataset_name}/pred.json")
    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)

    return args

def split_data(data):
    '''
    Split the data by the images number
    ex: {
        2: [sample1, ...]
        3: [sample2, ...]
    }
    '''
    data_dict = {}
    for d in data:
        n_img = len(d['task_instance']['images_path'])
        if n_img in data_dict:
            data_dict[n_img].append(d)
        else:
            data_dict[n_img] = [d]
    return data_dict
def main(args):
    import torch

    print(f'{datetime.now()}: Generating with model {args.model_name} on dataset {args.dataset_name}')

    # Load dataset
    dataset_dir = os.path.join(args.data_dir, args.dataset_name)
    img_dir = os.path.join(dataset_dir, 'images')
    dataset_json = f'{args.dataset_name}_combined_{args.combine_image}.json' if args.combine_image else f'{args.dataset_name}.json'
    dataset_path = os.path.join(dataset_dir, dataset_json)
    core_annotation = json.load(open(dataset_path))

    # Split data by number of images per sample
    data_dict = split_data(core_annotation['data'])

    # Initialize model worker
    worker_class = get_worker_class(args.model_name)
    models_configs = OmegaConf.load(args.model_configs)
    config = models_configs[args.model_name]
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    worker = worker_class.from_config(config=config)
    worker.model.to(config.device)
    worker.model.eval()

    prediction_results = []

    for n_img, sub_data in data_dict.items():
        # LIMIT to a few examples

        print(f'Processing samples with {n_img} images | Using {len(sub_data)} samples')

        lc_dataset = MileBenchDataset(
            annotation=sub_data,
            task_instructions=core_annotation['meta_data']['task_instruction'],
            img_dir=img_dir,
            max_context_len=config.max_context_len,
            n_tokens_per_image=config.n_tokens_per_image,
            tokenizer=worker.tokenizer,
            dataset_name=args.dataset_name,
            combine_image=args.combine_image,
        )

        dataloader = DataLoader(
            dataset=lc_dataset,
            batch_size=max(int(args.batch_image / n_img), 1),
            shuffle=False,
            num_workers=0,
            collate_fn=lc_dataset.collate_fn
        )

        # Perform inference
        for batch in tqdm(dataloader):
            with torch.no_grad():
                outputs = worker(device=config.device, **batch)
                prediction_results.extend(outputs)

    # Save result
    json.dump(prediction_results, open(args.output_pth, 'w'), ensure_ascii=False, indent=4)
    print(f"Saved predictions to {args.output_pth}")

if __name__ == '__main__':
    args = parse_args()
    main(args)