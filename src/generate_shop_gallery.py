from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, 'image', img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta, 'img_path' : img_path}

  def __len__(self):
    return len(self.images)

#Function to perform prefetch test
def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  det_results = []
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']

    #Generate format
    det_result = generate_format(img_id, pre_processed_images['img_path'], ret, opt.vis_thresh)

    #print(det_result)

    det_results.append(det_result)

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()

  return det_results


#Function to generate result format
def generate_format(image_id, image_path, results, threshold):
  bbox_list = []
  class_list = []
  conf_score_list = []

  for j in range(1, len(results['results']) + 1):
    for bbox in results['results'][j]:
      if bbox[4] > threshold:
        bbox_list.append(bbox[:4])
        class_list.append(j)
        conf_score_list.append(bbox[4])

  item = {'gallery_image_id' : int(image_id),
          'image_path' : str(image_path),
          'gallery_bbox' : bbox_list,
          'gallery_class' : class_list,
          'query_score' : conf_score_list,
  }

  return item

opt = opts().parse()
det_results = prefetch_test(opt)

import pickle

phase = 'test' if opt.test else 'val'

if opt.consumer_gallery:
  pickle_file = './consumer_gallery_{}.pkl'.format(phase)
else:
  pickle_file = './shop_gallery_{}.pkl'.format(phase)

with open(pickle_file, 'wb') as f:
  pickle.dump(det_results, f)

print('Complete...')