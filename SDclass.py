import os, gc, requests, random, shutil
import torch
import clip
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from IPython.utils.io import capture_output
from pycocotools.coco import COCO
from pytorch_fid import fid_score
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from DeepCache import DeepCacheSDHelper
from tgate import TgateSDLoader, TgateSDXLLoader, TgateSDDeepCacheLoader, TgateSDXLDeepCacheLoader
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
  # Initializations
  # =============================================================================
  def __init__(self, scheduler_dict, cache_model="deepcache", model='SD', clip_model='ViT-B/32', data_path='img_data', device=None, use_coco_imgs=False, N_clip=512, N_fid=0):
    '''
    Initializes Stable Diffusion pipeline with scheduler and cache model
    cache_model is a string with possible values: "tgate", "deepcache", "both" or None
    scheduler_dict is a dictionary with keys 'scheduler', 'params' (and 'name' for FID calculation)
    (scheduler_name, cache_model and model are also used for naming generated images folder for FID)
    '''
    self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.clip_model = clip_model
    self.data_path = data_path
    self.use_coco_imgs = use_coco_imgs

    self.num_inference_steps = 15
    self.img_ids = None
    self.coco_imgs = None
    self.coco_prompts = None
    
    os.makedirs(self.data_path, exist_ok=True)

    self.init_pipe(model)
    self.init_scheduler(scheduler_dict)
    self.init_cacher(cache_model, init=True)
    self.init_CLIP_model(clip_model)
    self.init_COCO_data(N_clip=N_clip, N_fid=N_fid, use_coco_imgs=use_coco_imgs)

    
  
  def init_pipe(self, model):
    '''
    Initializes Stable Diffusion pipeline
    works only for model SD or SDXL
    '''
    self.model = model
    with capture_output():
      if self.model == "SD":
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
      elif self.model == "SDXL":
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
      else:
        raise ValueError(f"Unknown model {self.model}")
    
    self.pipe = pipe.to(self.device)
    self.pipe.set_progress_bar_config(disable=True)

  def init_scheduler(self, scheduler_dict):
    '''
    Initializes scheduler
    '''
    self.scheduler_dict = scheduler_dict
    self.pipe.scheduler = scheduler_dict['scheduler'].from_config(
      self.pipe.scheduler.config, 
      **scheduler_dict.get('params', {})
      )

  def init_CLIP_model(self, clip_model):
    '''
    Initializes CLIP model
    '''
    self.clip_model = clip_model
    self.clip_model,  self.clip_preprocess = clip.load(clip_model)
    self.clip_model = self.clip_model.to(self.device).eval()
      
  def init_cacher(self, cache_model, init=False):
    '''
    Initializes cache model based on self.cache_model
    Possible values: "tgate", "deepcache", "both", None
    '''
    self.cache_model = cache_model
    if not init:
      self.init_pipe(self.model)

    tgate_both_loaders = dict(
        tgate = {
          'SD'  : TgateSDLoader,
          'SDXL': TgateSDXLLoader
        },
        both = {
          'SD'  : lambda x: TgateSDDeepCacheLoader(x, 3, 0),
          'SDXL': lambda x: TgateSDXLDeepCacheLoader(x, 3, 0)
        }
      )

    if self.cache_model == "deepcache":
      helper = DeepCacheSDHelper(pipe=self.pipe)
      helper.set_params(cache_interval=3,cache_branch_id=0)
      helper.enable()

    elif self.cache_model in ['tgate', 'both']:
      self.pipe = tgate_both_loaders[self.cache_model][self.model](self.pipe)

    elif self.cache_model is not None:
      raise ValueError(f"Unknown cache model {self.cache_model}")

  # =============================================================================
  # COCO dataset
  # =============================================================================
  def _is_image_ok(self, image_path):
      try: 
        with Image.open(image_path) as img: 
          return img.size == (299,299) 
      except: 
        return False
      
  def init_COCO_data(self, N_clip, N_fid, path_coco_imgs=None, use_coco_imgs=False):
    '''
    Downloads and extracts MSCOCO dataset with annotations and images
    If use_coco_imgs is False will not download images
    Sets validation and test image ids
    '''
    self.use_coco_imgs = use_coco_imgs
    self.path_coco_imgs = path_coco_imgs or os.path.join(self.data_path, 'imgs_coco')
    os.makedirs(self.path_coco_imgs, exist_ok=True)

    if not os.path.exists(os.path.join(self.data_path, 'annotations')):
      annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
      annotations_path = 'annotations_trainval2017.zip'
      response = requests.get(annotations_url)
      open(annotations_path, 'wb').write(response.content)
      import zipfile
      with zipfile.ZipFile(annotations_path, 'r') as zip_ref:
        zip_ref.extractall(self.data_path)
      os.remove(annotations_path)
    
    # supress useless coco output
    with capture_output(True, False, True):
      self.coco_imgs = self.coco_imgs or COCO(os.path.join(self.data_path, 'annotations/instances_train2017.json'))
      self.coco_prompts = self.coco_prompts or COCO(os.path.join(self.data_path, 'annotations/captions_train2017.json'))
      img_ids = self.coco_imgs.getImgIds()

      random.seed(42)
      random.shuffle(img_ids)
      self.img_ids = {'clip': img_ids[0:N_clip], 'fid': img_ids[-N_fid:]}

      # download images
      if use_coco_imgs:
        already_downloaded = set(os.listdir(self.path_coco_imgs))
        already_downloaded = set([img_name for img_name in already_downloaded if self._is_image_ok(f"{self.path_coco_imgs}/{img_name}")])

        images = self.coco_imgs.loadImgs(self.img_ids['fid'])
        for img in tqdm(images, desc='downloading imgs'):
          if f"{img['id']}.png" in already_downloaded: continue
          img_url = img['coco_url']
          img_data = requests.get(img_url).content
          
          with open(f"{self.path_coco_imgs}/{img['id']}.png", 'wb') as handler:
            handler.write(img_data)
          
          # resize images to 299x299
          img_coco = Image.open(f"{self.path_coco_imgs}/{img_id}.png")
          img_coco = img_coco.resize((299, 299), Image.LANCZOS)
          img_coco.save(f"{self.path_coco_imgs}/{img_id}.png")

  
  # =============================================================================
  # __call__
  # =============================================================================
  def __call__(self, prompt, **kwargs):
    '''
    Returns generated image by prompt based on which cache model is used
    '''
    call_params = dict(
        prompt = prompt,
        num_inference_steps = self.num_inference_steps,
    )

    if self.cache_model in [None, "deepcache"]:
      call_params.update(kwargs)
      return self.pipe(**call_params).images[0]

    call_params['gate_step'] = max(call_params['num_inference_steps']//2.5, 1)
    call_params.update(kwargs)
    return self.pipe.tgate(**call_params).images[0]
  
  # =============================================================================
  # Tflops
  # =============================================================================
  def Tflops(self, prompt, **kwargs):
    '''
    Returns GFLOPS depending on the cache_model, scheduler, inference_steps
    '''
    torch.manual_seed(kwargs.get("seed", 42))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
      with record_function("model_inference"):
        self(prompt, **kwargs)

    # if self.verbose:
    #   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    total_flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])
    return round(total_flops/1e12,3)
  

  # =============================================================================
  # CLIP
  # =============================================================================
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
  
  def CLIP(self, **pipe_kwargs):
    '''
    Generates conditional images and calculates CLIP on MSCOCO dataset
    '''
    clip_scores = []
    for img_id in tqdm(self.img_ids['clip'], desc=f"CLIP_{self.num_inference_steps}"):
      torch.manual_seed(img_id)
      random.seed(img_id)

      ann_ids = self.coco_prompts.getAnnIds(imgIds=img_id)
      prompts = self.coco_prompts.loadAnns(ann_ids)
      prompt = random.choice([ann['caption'] for ann in prompts])
      img_gen_cond = self(prompt, **pipe_kwargs)

      clip_gen  = self._get_clip_score(img_gen_cond, prompt)
      clip_scores.append(float(clip_gen))

    return np.mean(clip_scores)


  # =============================================================================
  # FID
  # =============================================================================
  def FID(self, path_gen=None, delete_gen_after=False, **pipe_kwargs):
    '''
    Generates unconditional small images and calculates FID on resized MSCOCO dataset
    if delete_gen_after is True, generated images will be deleted after FID calculation
    '''
    if not self.use_coco_imgs:
      raise ValueError("first self.init_COCO_data(..., use_coco_imgs=True) must be applied") 
    
    if path_gen==None: 
      path_gen = f'imgs_{self.model}/cache_{self.cache_model}/{self.scheduler_dict["name"]}/{self.num_inference_steps}'
      path_gen = os.path.join(self.data_path, path_gen)
    self.path_gen = path_gen
    os.makedirs(path_gen, exist_ok=True)

    # gen & resize loop:
    already_generated = os.listdir(self.path_gen)
    already_generated = set([img_name for img_name in already_generated if self._is_image_ok(f"{self.path_gen}/{img_name}")])
    img_ids_to_process = [img_id for img_id in self.img_ids['fid'] if  f"{img_id}.png" not in already_generated]
    
    for img_id in tqdm(img_ids_to_process, desc=f"FID_{self.num_inference_steps}"):
      torch.manual_seed(img_id)
      img_gen_uncond = self("", guidance_scale = 1.0, **pipe_kwargs)
      img_gen_uncond = img_gen_uncond.resize((299, 299), Image.LANCZOS)
      img_gen_uncond.save(f"{self.path_gen}/{img_id}.png")
    
    # FID stat

    # make sure all files ok:
    for img_name in os.listdir(self.path_coco_imgs):
      if int(img_name.split('.')[0]) not in self.img_ids['fid']:
        os.remove(f"{self.path_coco_imgs}/{img_name}")
    for img_name in os.listdir(self.path_gen):
      if int(img_name.split('.')[0]) not in self.img_ids['fid']:
        os.remove(f"{self.path_gen}/{img_name}")
    assert len(os.listdir(self.path_coco_imgs)) == len(self.img_ids['fid'])
    assert len(os.listdir(self.path_gen)) == len(self.img_ids['fid'])

    fid_params = {'batch_size': 16, 'num_workers': 0, 'device': self.device, 'dims': 2048}
    fid_value = fid_score.calculate_fid_given_paths([self.path_coco_imgs, self.path_gen], **fid_params)

    if delete_gen_after:
      shutil.rmtree(self.path_gen)

    return fid_value
  
  # =============================================================================
  # Combined stats
  # =============================================================================
  def STATS(self, list_inference_steps, get_fid=False, pipe_kwargs_list=[]):
    pipe_kwargs_list = pipe_kwargs_list or [{} for _ in list_inference_steps]
    if get_fid and not self.use_coco_imgs:
      raise ValueError("first self.init_COCO_data(..., use_coco_imgs=True) must be applied") 
    init_inference_steps = self.num_inference_steps
    Tflops = []
    Clips  = []
    Fids   = []
    for i, steps in enumerate(list_inference_steps):
      self.num_inference_steps = steps
      
      if get_fid:
        fid = self.FID(**pipe_kwargs_list[i])
        Fids.append(round(fid,3))
      else:
        Fids.append("None")

      Tflops_cond = self.Tflops(prompt="a photograph of an astronaut riding a horse")
      Tflops.append(round(Tflops_cond, 3))

      clip_mean = self.CLIP(**pipe_kwargs_list[i])
      Clips.append(round(clip_mean,3))

    # if self.verbose:
    #   header = f"\n{'Inference Steps':<20} {'TFlops':<15} {'Clip':<15} {'FID':<10}"
    #   print(header)
    #   print("-" * len(header))
    #   for i, steps in enumerate(list_inference_steps):
    #     print(f"{steps:<20} {Tflops[i]:<15} {Clips[i]:<15} {Fids[i]:<10}")

    self.num_inference_steps = init_inference_steps
    return Tflops, Clips, Fids