import os, gc, requests, random, shutil
import torch
import clip
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from pycocotools.coco import COCO
from pytorch_fid import fid_score
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from DeepCache import DeepCacheSDHelper
from tgate import TgateSDLoader, TgateSDXLLoader
from torch.profiler import profile, record_function, ProfilerActivity

class SDCompare:
  '''
  Class for gathering CLIP and FID statistics for Stable Diffusion 2.1
  depending on which scheduler, cache model and inference steps are used

  mscoco: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
  CLIP: https://github.com/Taited/clip-score/blob/master/src/clip_score/clip_score.py
  FID: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
  '''

  # =============================================================================
  # Initialization
  # =============================================================================
  def __init__(self, scheduler_dict, cache_model="deepcache", model='SD', clip_model='ViT-B/32', data_path='img_data', device=None):
    '''
    Initializes Stable Diffusion pipeline with scheduler and cache model
    cache_model is a string with possible values: "tgate", "deepcache", "both" or None
    scheduler_dict is a dictionary with keys 'scheduler', 'params' and 'name'
    (scheduler_name, cache_model and model are also used for naming generated images folder)
    '''
    self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = model
    self.cache_model = cache_model
    self.DeepCacheHelper = None
    self.scheduler_dict = scheduler_dict
    self.clip_model = clip_model

    self.data_path = data_path
    os.makedirs(self.data_path, exist_ok=True)
    
    self.init_pipe()
    self.init_scheduler()
    self.init_cacher()
    self.init_CLIP_model()
    self.init_COCO_data()
    
    self.inference_steps = 15
  
  def init_pipe(self, model=None):
    '''
    Initializes Stable Diffusion pipeline
    '''
    model = model or self.model
    self.model = model

    if self.model == "SD":
      pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    elif self.model == "SDXL":
      pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    else:
      raise ValueError(f"Unknown model {self.model}")
    self.pipe = pipe.to(self.device)
    self.pipe.set_progress_bar_config(disable=True)

  def init_scheduler(self, scheduler_dict=None):
    '''
    Initializes scheduler
    '''
    scheduler_dict = scheduler_dict or self.scheduler_dict
    self.scheduler_dict = scheduler_dict

    self.pipe.scheduler = scheduler_dict['scheduler'].from_config(self.pipe.scheduler.config, **scheduler_dict.get('params', {}))

  def init_CLIP_model(self, clip_model=None):
    '''
    Initializes CLIP model
    '''
    clip_model = clip_model or self.clip_model
    self.clip_model = clip_model

    self.clip_model,  self.clip_preprocess = clip.load(clip_model)
    self.clip_model = self.clip_model.to(self.device).eval()
      
  def init_cacher(self, cache_model=None):
    '''
    Initializes cache model based on self.cache_model
    Possible values: "tgate", "deepcache", "both"
    '''
    cache_model = cache_model or self.cache_model
    self.cache_model = cache_model

    if self.cache_model in ["tgate", "both"]:
      if self.model == "SD":
        self.pipe = TgateSDLoader(self.pipe).to(self.device)
      elif self.model == "SDXL":
        self.pipe = TgateSDXLLoader(self.pipe).to(self.device)
      else:
        raise ValueError(f"Unknown model {self.model}")
    if self.cache_model in ["deepcache", "both"]:
      self.DeepCacheHelper = DeepCacheSDHelper(pipe=self.pipe)
      self.DeepCacheHelper.set_params(
          cache_interval=3,
          cache_branch_id=0,
      )
      self.DeepCacheHelper.enable()
    elif self.cache_model == 'tgate':
      if self.DeepCacheHelper!=None:
        self.DeepCacheHelper.disable()
    elif self.cache_model is not None:
      raise ValueError(f"Unknown cache model {self.cache_model}")
  
  def init_COCO_data(self, N_val=512, N_test=1024, path_coco_imgs=None, path_coco_FID=None):
    '''
    Downloads and extracts MSCOCO dataset with annotations and images
    Sets validation and test image ids
    '''
    self.path_coco_imgs = path_coco_imgs or os.path.join(self.data_path, 'imgs_coco')
    self.path_coco_FID = path_coco_FID or os.path.join(self.data_path, 'imgs_coco_FID')
    os.makedirs(self.path_coco_imgs, exist_ok=True)
    os.makedirs(self.path_coco_FID, exist_ok=True)

    if not os.path.exists(os.path.join(self.data_path, 'annotations')):
      annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
      annotations_path = 'annotations_trainval2017.zip'
      response = requests.get(annotations_url)
      open(annotations_path, 'wb').write(response.content)
      import zipfile
      with zipfile.ZipFile(annotations_path, 'r') as zip_ref:
        zip_ref.extractall(self.data_path)
      os.remove(annotations_path)
    
    
    self.coco_imgs = COCO(os.path.join(self.data_path, 'annotations/instances_train2017.json'))
    self.coco_prompts = COCO(os.path.join(self.data_path, 'annotations/captions_train2017.json'))
    img_ids = self.coco_imgs.getImgIds()

    random.seed(42)
    random.shuffle(img_ids)
    self.img_ids = {'val': img_ids[0:N_val], 'test': img_ids[-N_test:]}

    # download images
    print('downloading images...')
    already_downloaded = os.listdir(self.path_coco_imgs)
    images = self.coco_imgs.loadImgs(self.img_ids['val'] + self.img_ids['test'])
    for img in tqdm(images):
      if f"{img['id']}.png" in already_downloaded:
        continue
      img_url = img['coco_url']
      img_data = requests.get(img_url).content
      
      with open(f"{self.path_coco_imgs}/{img['id']}.png", 'wb') as handler:
        handler.write(img_data)
    
    # copy and resize images to 299x299
    already_downloaded = os.listdir(self.path_coco_FID)
    for img_id in tqdm(self.img_ids['val'] + self.img_ids['test']):
      if f"{img_id}.png" in already_downloaded:
        continue
      img_coco = Image.open(f"{self.path_coco_imgs}/{img_id}.png")
      img_coco = img_coco.resize((299, 299), Image.LANCZOS)
      img_coco.save(f"{self.path_coco_FID}/{img_id}.png")

  
  # =============================================================================
  # __call__ and Utilities
  # =============================================================================
  def __call__(self, prompt, **kwargs):
    '''
    Returns generated image by prompt based on which cache model is used
    '''
    call_params = dict(
        prompt = prompt,
        num_inference_steps = self.inference_steps,
    )

    if self.cache_model == "deepcache":
      call_params.update(kwargs)
      return self.pipe(**call_params).images[0]

    call_params['gate_step'] = max(call_params['num_inference_steps']//2.5, 1)
    call_params.update(kwargs)
    return self.pipe.tgate(**call_params).images[0]

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
  
  # =============================================================================
  # Tflops
  # =============================================================================
  def Tflops(self, prompt, print_table=False, **kwargs):
    '''
    Returns GFLOPS depending on the cache_model, scheduler, inference_steps
    '''
    torch.manual_seed(kwargs.get("seed", 42))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
      with record_function("model_inference"):
        self(prompt, **kwargs)

    if print_table:
      print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    total_flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])
    return round(total_flops/1e12,3)
  

  # =============================================================================
  # CLIP
  # =============================================================================
  def CLIP(self, val_test='val'):
    '''
    Generates conditional images and calculates CLIP on MSCOCO dataset
    '''
    clip_scores = torch.zeros((len(self.img_ids[val_test]), 2), dtype=torch.float32)
    # gen loop:
    for n, img_id in enumerate(tqdm(self.img_ids[val_test], desc="CLIP")):
      torch.manual_seed(n)
      random.seed(n)

      ann_ids = self.coco_prompts.getAnnIds(imgIds=img_id)
      prompts = self.coco_prompts.loadAnns(ann_ids)
      prompt = random.choice([ann['caption'] for ann in prompts])

      img_gen_cond = self(prompt)
      img_coco = Image.open(f"{self.path_coco_imgs}/{img_id}.png")

      clip_gen  = float(self._get_clip_score(img_gen_cond, prompt))
      clip_real = float(self._get_clip_score(img_coco, prompt))
      clip_scores[n] = torch.tensor([clip_gen, clip_real])

    # CLIP stats
    clip_mean = clip_scores[:,0].mean()
    clip_diff = (clip_scores[:,0]-clip_scores[:,1]).abs().mean()
    return float(clip_mean), float(clip_diff)


  # =============================================================================
  # FID
  # =============================================================================
  def FID(self, val_test='val', path_gen=None, delete_gen_after=True, **fid_kwargs):
    '''
    Generates unconditional small images and calculates FID on resized MSCOCO dataset
    if delete_gen_after is True, generated images will be deleted after FID calculation
    '''
    if path_gen==None: 
      path_gen = f'imgs_{self.model}/cache_{self.cache_model}/{self.scheduler_dict["name"]}/{self.inference_steps}'
      path_gen = os.path.join(self.data_path, path_gen)
    self.path_gen = path_gen
    self.path_gen_FID  = os.path.join(path_gen,  'FID')
    os.makedirs(path_gen, exist_ok=True)
    os.makedirs(self.path_gen_FID,  exist_ok=True)

    # gen & resize loop:
    already_generated = os.listdir(self.path_gen_FID)
    for n, img_id in enumerate(tqdm(self.img_ids[val_test], desc="FID")):
      if f"{img_id}.png" in already_generated:
        continue
      torch.manual_seed(n)
      img_gen_uncond = self("")
      img_gen_uncond = img_gen_uncond.resize((299, 299), Image.LANCZOS)
      img_gen_uncond.save(f"{self.path_gen_FID}/{img_id}.png")
    
    # FID stat
    fid_params = {'batch_size': 16, 'num_workers': 1, 'device': self.device, 'dims': 2048}
    fid_params.update(fid_kwargs)
    fid_value = fid_score.calculate_fid_given_paths([self.path_coco_FID, self.path_gen_FID], **fid_params)

    if delete_gen_after:
      shutil.rmtree(self.path_gen_FID)

    return fid_value
  
  # =============================================================================
  # Combined stats
  # =============================================================================
  def stats(self, num_inference_steps=None, get_fid=False):
    num_inference_steps = num_inference_steps or self.num_inference_steps
    self.num_inference_steps = num_inference_steps

    Tflops_cond = self.Tflops(prompt="a photograph of an astronaut riding a horse")
    print(f"Tflops: {Tflops_cond:3}\n")

    clip_mean, clip_diff = self.CLIP()
    print(f"CLIP_mean: {clip_mean:3}, CLIP_diff: {clip_diff:3}\n")

    if get_fid:
      fid = self.FID()
      print(f"FID: {fid:3}\n")