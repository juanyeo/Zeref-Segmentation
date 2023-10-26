from detectron2.data import MetadataCatalog, DatasetMapper, build_detection_train_loader, build_detection_test_loader
from data_loader.refcoco_mapper import RefCOCOMapper
from detectron2.config import get_cfg
import torch
import numpy as np
import os
from models.CLIPSeg import CLIPSeg
from utils.visualization import show_image_seg
#import data_loader.register_refcoco

def build_test_loader(cfg, dataset_name):
    #assert cfg.INPUT.DATASET_MAPPER_NAME == "refcoco"
    #mapper = RefCOCOMapper(cfg, False)
    #return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, is_train=False), batch_size=8)

if __name__ == "__main__":
    cfg = get_cfg()
    test_loader = build_test_loader(cfg, "refcocog_google_val")
    #test_data = next(iter(test_loader))

    '''
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    image = Image.open(test_data[0]['file_name']).convert("RGB")
    images = [preprocess(image)]
    image_input = torch.tensor(np.stack(images)).cuda()
    text_tokens = clip.tokenize([test_data[0]['sentence']['raw']]).cuda()

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    print(similarity)
    '''
    
    N = 20
    test_iter = iter(test_loader)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = CLIPSeg()

    for i in range(N):
        print('[%d]' % i)
        test_data = next(test_iter)
        # Patch-wise Segmentation
        prediction = model.predict(test_data[0]['file_name'], test_data[0]['sentence']['raw'])
        # Show Text
        print("Text : " + test_data[0]['sentence']['raw'])
        # Show Image
        show_image_seg(test_data[0]['file_name'], prediction, 'BuPu', 'result'+str(i))
 
