import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/FFHQ.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    N = 100
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], n_timestep=N, schedule_phase='val')  # 100

    logger.info('Begin Model Inference.')

    # average
    for i in range(1):
        # step = i
        index = 1

        current_step = 0
        current_epoch = 0
        idx = 0

        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        avg_psnr = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_niqe = 0.0
        total_lpips = 0.0

        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)

            # test
            single_psnr, single_ssim, single_niqe, single_lpips, img_name = diffusion.test_PPO(continous=False)
            total_psnr += single_psnr
            total_ssim += single_ssim
            total_niqe += single_niqe
            total_lpips += single_lpips

            visuals = diffusion.get_current_visuals(need_LR=True)

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}'.format(result_path, img_name))

        print("n_timestep: ", N, "avg_psnr: ", total_psnr / idx,
                  "avg_ssim: ", total_ssim / idx,
                  "avg_niqe: ", total_niqe / idx, "avg_lpips: ", total_lpips / idx)

