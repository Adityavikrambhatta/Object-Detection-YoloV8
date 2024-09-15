import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os 
from src import config

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.data_yml = config.get_config(f"data_set/data.yaml")

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = [img['id'] for img in self.annotations['images']]
        self.images = [img for img in self.annotations['images']]
        self.annotations = [ annot for annot in self.annotations['annotations']]

    def __len__(self):
        return len(self.image_ids)
    
    def __generatelabeltxt__(self, mode):

         # Create output directory if it doesn't exist
        
        print(self.root_dir)

        label_dir = f'{self.root_dir}/labels/{mode}/'

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

            images = self.images
            annotations = self.annotations
            print(len(images))
            print(len(annotations))

            for i, image in enumerate(images) :
                for annot in annotations:
                    if   image['id'] == annot['image_id']:
                        file_name = image['file_name']
                        bbox = annot['bbox']
                        height = image['height']
                        width = image['width']

                        cid = annot['category_id'] - 1
                        cname = self.data_yml['names'][cid]
                        c1, c2, c3, c4 = bbox
                        x_max = c1 + c3
                        y_max = c2 + c4
                        xc = (c1 + x_max) / 2
                        yc = (c2 + y_max) / 2 

                        # Prepare the content for the text file
                        annotation_str = f'{cid} {xc/width} {yc/height} {c3/width} {c4/height}\n'
                        # if label_files.get(image["id"]) :
                        #     label_files[image["id"]]["content"] += annotation_str
                        # else :
                        #     label_files.update({image["id"] : { "file_name" : file_name, "content" : annotation_str }})
                        txt_file_path = os.path.join(label_dir, file_name.replace('.jpg', '.txt'))
                        with open(txt_file_path, 'a') as txt_file:
                            txt_file.write(annotation_str)
                    

def create_datasets(cfg, mode):
    # Implement train/val split logic here
    # For simplicity, we're using the same dataset for bofth train and val
    wk_dir = f"{os.getcwd()}/{cfg['data']}/"
    print(wk_dir)
    dataset = COCODataset( wk_dir, f"{wk_dir}/{cfg['annotations_file']}" )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataset = dataset.__generatelabeltxt__( 'train') 
    val_dataset = dataset.__generatelabeltxt__( 'valid')
    return train_dataset, val_dataset



