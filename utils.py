
import os, requests, random, pickle, zipfile
import numpy as np
import torch, clip
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pytorch_fid import fid_score
from torch.profiler import profile, record_function, ProfilerActivity

# ==================================================================================================
# other
# ==================================================================================================
def disable_pipe_bar(f):
  '''
  decorator which disables pipe progress bar
  '''
  def wrapper(*args, **kwargs):
    args[0].model.set_progress_bar_config(disable=True)
    res = f(*args, **kwargs)
    args[0].model.set_progress_bar_config(disable=False)
    return res
  return wrapper

# ==================================================================================================
# COCO ANNOTATIONS & IMGS
# ==================================================================================================
COCO_ANNOTATIONS = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_IMGS = 'annotations/instances_train2017.json'
COCO_ANNS = 'annotations/captions_train2017.json'

def download_COCO(N_ann, N_fid, path_coco='coco_data', seed=42):
  '''
  Downloads and extracts MSCOCO dataset including N_fid images
  Return N_ann annotations
  '''
  path_coco_imgs = os.path.join(path_coco, 'imgs_coco')
  os.makedirs(path_coco, exist_ok=True)
  os.makedirs(path_coco_imgs, exist_ok=True)
  
  # download annotations & img_ids
  if not os.path.exists(os.path.join(path_coco, 'annotations')):
    _temp_zip = 'annotations_trainval2017.zip'
    with open(_temp_zip, 'wb') as f:
      f.write(requests.get(COCO_ANNOTATIONS).content)
    with zipfile.ZipFile(_temp_zip, 'r') as zip_ref:
      zip_ref.extractall(path_coco)
    os.remove(_temp_zip)
  
  coco_imgs = COCO(os.path.join(path_coco, COCO_IMGS))
  img_ids_path = os.path.join(path_coco, 'img_ids.pkl')

  if os.path.exists(img_ids_path):
    img_ids = pickle.load(open(img_ids_path, 'rb'))
  else:
    img_ids = coco_imgs.getImgIds()
    if seed is not None:
      random.seed(seed)
      random.shuffle(img_ids)
    img_ids = img_ids[0:N_ann+N_fid]

    with open(img_ids_path, 'wb') as f:
      pickle.dump(img_ids, f)

  # load annotations
  coco_prompts = COCO(os.path.join(path_coco, COCO_ANNS))
  ann_ids = coco_prompts.getAnnIds(imgIds=img_ids[:N_ann])
  prompts = coco_prompts.loadAnns(ann_ids)

  # download images
  already_downloaded = set(os.listdir(path_coco_imgs))
  imgs_to_download = coco_imgs.loadImgs(img_ids[-N_fid:])
  imgs_to_download = [img for img in imgs_to_download if f"{img['id']}.png" not in already_downloaded]
  for img in tqdm(imgs_to_download, desc='Downloading images'):
    img_url = img['coco_url']
    img_data = requests.get(img_url).content

    with open(os.path.join(path_coco_imgs, f"{img['id']}.png"), 'wb') as f:
      f.write(img_data)
  
  # resize all images to 299x299
  for img_id in tqdm(img_ids[-N_fid:], desc='Resizing images'):
    img_path = os.path.join(path_coco_imgs, f"{img_id}.png")
    with Image.open(img_path) as img:
      img.resize((299, 299), Image.LANCZOS).save(img_path)
  
  return prompts

def validate_coco_images(path_coco_imgs='coco_data/imgs_coco'):
  '''
  Validate coco images, and if image is broken delete it
  '''
  all_imgs = os.listdir(path_coco_imgs)
  for img in tqdm(all_imgs, desc='Validating images'):
    img_path = os.path.join(path_coco_imgs, img)
    try:
      with Image.open(img_path) as img:
        if img.size != (299, 299):
          img.resize((299, 299), Image.LANCZOS).save(img_path)
    except:
      os.remove(img_path)

# ==================================================================================================
# Tflops, CLIP & FID EVALUATOR
# ==================================================================================================
class SDEvaluator:
  '''
  Class for gathering CLIP and FID statistics for Diffusion pipeline
  '''
  def __init__(self, model, device = None, clip_model = 'ViT-B/32'):
    self.model = model
    self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.clip_model,  self.clip_preprocess = clip.load(clip_model)
    self.clip_model = self.clip_model.to(self.device).eval()
  
  def generate(self, *args, **kwargs):
    return self.model(*args, **kwargs).images[0]
  

  def _get_clip_score(self, image, caption):
    '''
    Returns CLIP score for one image and one caption
    see https://github.com/Taited/clip-score/blob/master/src/clip_score/clip_score.py
    '''
    image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
    text_input  = clip.tokenize([caption]).to(self.device)

    with torch.no_grad():
      image_features = self.clip_model.encode_image(image_input)
      text_features  = self.clip_model.encode_text(text_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features / text_features.norm(dim=-1, keepdim=True)

    clip_score = torch.matmul(image_features, text_features.T).item()
    return clip_score

  @disable_pipe_bar
  def CLIP(self, annotations, path_gen_img=None, verbose=True, **kwargs):
    '''
    Generates conditional images and calculates CLIP on MSCOCO dataset
    '''
    clip_scores = []
    n_generated = len(os.listdir(path_gen_img)) if os.path.exists(path_gen_img) else 0
    for i, prompt in tqdm(enumerate(annotations[n_generated:]), desc="CLIP", total=len(annotations), disable = not verbose):
      torch.manual_seed(i)
      img_gen_cond = self.generate(prompt, **kwargs) 
      clip_gen = self._get_clip_score(img_gen_cond, prompt)
      if path_gen_img is not None: clip_gen.save(os.path.join(path_gen_img, f"{i}.png"))
      clip_scores.append(float(clip_gen))
    return np.mean(clip_scores)
  
  @disable_pipe_bar
  def FID(self, path_gen_img, path_true_img, verbose=True, **kwargs):
    '''
    Generates unconditional small images and calculates FID
    if delete_gen_after is True, generated images will be deleted after FID calculation
    '''
    os.makedirs(path_gen_img, exist_ok=True)

    # unconditional generation
    default_kwargs = dict(
      prompt = "",
      guidance_scale = 1,
    )
    default_kwargs.update(kwargs)

    # gen & resize loop:
    n_true_img  = len(os.listdir(path_true_img))
    n_generated = len(os.listdir(path_gen_img))
    for i in tqdm(range(n_generated, n_true_img), desc="FID", disable=not verbose):
      torch.manual_seed(i)
      img_gen_uncond = self.generate(**default_kwargs)
      img_gen_uncond = img_gen_uncond.resize((299, 299), Image.LANCZOS)
      img_gen_uncond.save(os.path.join(path_gen_img, f"{i}.png"))
    
    # FID stat
    fid_params = {'batch_size': 16, 'num_workers': 1, 'device': self.device, 'dims': 2048}
    fid_value = fid_score.calculate_fid_given_paths([path_true_img, path_gen_img], **fid_params)
    return fid_value
  
  @disable_pipe_bar
  def Tflops(self, prompt=None, **kwargs):
    '''
    Count Tflops
    '''
    prompt = prompt or "a photograph of an astronaut riding a horse"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
      with record_function("model_inference"):
        self.generate(prompt, **kwargs)
    
    total_flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])
    return round(total_flops/1e12,3)