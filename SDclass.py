import os, gc, requests, random
import torch
import clip
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from pycocotools.coco import COCO
from pytorch_fid import fid_score
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from DeepCache import DeepCacheSDHelpe
from tgate import TgateSDLoader
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
  def __init__(self, scheduler, scheduler_name, cache_model="deepcache", model='SD'):
    '''
    Initializes Stable Diffusion pipeline with scheduler and cache model
    (scheduler_name and cache_model are also used for naming generated images folder)
    '''
    self.model = model
    self.cache_model = cache_model
    self.scheduler = {'scheduler': scheduler, 'name': scheduler_name}
    
    self.init_pipe()
    self.init_scheduler()
    self.init_cacher()
    self.init_COCO_annotations()
    self.init_CLIP_model()
    
    self.inference_steps = 15
    self.gen_height = 512
    self.gen_width  = 512
  
  def init_pipe(self):
    '''
    Initializes Stable Diffusion pipeline
    '''
    if self.model == "SD":
      pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    elif self.model == "SDXL":
      pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    self.pipe = pipe.to("cuda")

  def init_scheduler(self, scheduler=None):
    '''
    Initializes scheduler
    '''
    if scheduler==None: scheduler = self.scheduler
    self.scheduler = scheduler
    self.pipe.scheduler = scheduler

  def init_CLIP_model(self):
    '''
    Initializes CLIP model
    '''
    self.clip_model,  self.clip_preprocess = clip.load('ViT-B/32')
    self.clip_model = self.clip_model.to("cuda").eval()
      
  def init_cacher(self):
    '''
    Initializes cache model based on self.cache_model
    Possible values: "tgate", "deepcache", "both"
    '''
    if self.cache_model in ["tgate", "both"]:
      self.pipe = TgateSDLoader(self.pipe).to("cuda")
    if self.cache_model in ["deepcache", "both"]:
      helper = DeepCacheSDHelper(pipe=self.pipe)
      helper.set_params(
          cache_interval=3,
          cache_branch_id=0,
      )
      helper.enable()
  
  def init_COCO_annotations(self, N_val=2048, N_test=5000):
    '''
    Downloads and extracts MSCOCO dataset with annotations (not images yet)
    Sets validation and test image ids
    '''
    if not os.path.exists('annotations'):
      annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
      annotations_path = 'annotations_trainval2017.zip'
      response = requests.get(annotations_url)
      open(annotations_path, 'wb').write(response.content)
      import zipfile
      with zipfile.ZipFile(annotations_path, 'r') as zip_ref:
          zip_ref.extractall('.')
      os.remove(annotations_path)
    
    coco = COCO('annotations/instances_train2017.json')
    img_ids = coco.getImgIds()

    random.seed(42)
    random.shuffle(img_ids)
    self.img_ids = {'val': img_ids[0:N_val], 'test': img_ids[-N_test:]}
  
  
  # =============================================================================
  # Utilities
  # =============================================================================
  def __cal__(self, prompt, **kwargs):
    '''
    Returns generated image by prompt based on which cache model is used
    '''
    call_params = dict(
        prompt = prompt,
        num_inference_steps = self.inference_steps,
        height = self.gen_height,
        width  = self.gen_width,
    )

    if self.cache_model == "deepcache":
      call_params.update(kwargs)
      return self.pipe(**call_params).images[0]

    call_params['gate_step'] = max(self.inference_steps//2.5, 1)
    call_params.update(kwargs)
    return self.pipe.tgate(**call_params).images[0]
  
  def get_gflops(self, prompt, print_table=False, **kwargs):
    '''
    Returns GFLOPS depending on the cache_model, scheduler, inference_steps
    '''
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
      with record_function("model_inference"):
        self(prompt, **kwargs)

    if print_table:
      print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return round(prof.key_averages().flops/1e9,3)

  def get_clip_score(self, image, caption):
    '''
    Returns CLIP score for one image and one caption
    see https://github.com/Taited/clip-score/blob/master/src/clip_score/clip_score.py
    '''
    image_input = self.clip_preprocess(image).unsqueeze(0).to("cuda")
    text_input  = clip.tokenize([caption]).to("cuda")

    with torch.no_grad():
      image_features = self.clip_model.encode_image(image_input)
      text_features  = self.clip_model.encode_text(text_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features / text_features.norm(dim=-1, keepdim=True)

    clip_score = torch.matmul(image_features, text_features.T).item()
    return clip_score
  

  # =============================================================================
  # CLIP and FID
  # =============================================================================
  def get_stats(self, path_gen=None, path_coco='imgs_coco', val_test='test'):
    '''
    Generates conditional images and calculates CLIP on MSCOCO dataset
    Generates unconditional small images and calculates FID on resized MSCOCO dataset
    Returns stats dictionary for CLIP and FID
    '''

    if path_gen==None: 
      path_gen = f'imgs_SD/cache_{self.cache_model}/{self.scheduler["name"]}/{self.inference_steps}'

    os.mkdir(path_coco, exist_ok=True)
    os.mkdir(path_gen,  exist_ok=True)

    stats = {}
    assert(torch.cuda.is_available(), "cuda is not available")

    coco_already_exist = os.listdir(path_coco)
    coco_imgs = COCO('annotations/instances_train2017.json')
    coco_prompts = COCO('annotations/captions_train2017.json')

    clip_scores = torch.zeros((len(self.img_ids[val_test]), 2), dtype=torch.float32)

    # CLIP loop:
    for n, img_id in enumerate(tqdm(self.img_ids[val_test], desc="CLIP")):

      torch.manual_seed(n)
      random.seed(n)

      ann_ids = coco_prompts.getAnnIds(imgIds=img_id)
      prompts = coco_prompts.loadAnns(ann_ids)
      
      prompt = random.choice([ann['caption'] for ann in prompts])

      img_gen_cond = self(prompt)

      if img_id not in coco_already_exist:
        img_url = coco_imgs.loadImgs(img_id)[0]['coco_url']
        img_coco = Image.open(BytesIO(requests.get(img_url).content))
        img_coco.save(f"{path_coco}/{img_id}.png")
      else:
        img_coco = Image.open(f"{path_coco}/{img_id}.png")

        clip_gen  = float(self.get_clip_score(img_gen_cond, prompt))
        clip_real = float(self.get_clip_score(img_coco, prompt))
        clip_scores[n] = torch.tensor([clip_gen, clip_real])

    # CLIP stats
    stats['CLIP_mean'] = clip_scores[:,0].mean()
    stats['CLIP_diff'] = (clip_scores[:,0]-clip_scores[:,1]).abs().mean()

    path_coco_FID = os.path.join(path_coco, 'FID')
    path_gen_FID  = os.path.join(path_gen,  'FID')
    os.mkdir(path_coco_FID, exist_ok=True)
    os.mkdir(path_gen_FID,  exist_ok=True)
    coco_already_exist = os.listdir(path_coco_FID)

    gc.collect()
    torch.cuda.empty_cache()

    # FID loop:
    for n, img_id in enumerate(tqdm(self.img_ids[val_test], desc="FID")):
      torch.manual_seed(n)

      img_gen_uncond = self("", height=299, width=299)
      img_gen_uncond.save(f"{path_gen_FID}/{img_id}.png")

      if img_id not in coco_already_exist:
        img_coco = Image.open(f"{path_coco}/{img_id}.png")
        img_coco = img_coco.resize((299, 299), Image.ANTIALIAS)
        img_coco.save(f"{path_coco_FID}/{img_id}.png")
      else:
        img_coco = Image.open(f"{path_coco_FID}/{img_id}.png")
    
    # FID stat
    fid_value = fid_score.calculate_fid_given_paths([path_coco, path_gen], batch_size=50, device='cuda', dims=2048)
    stats['FID'] = fid_value

    return stats