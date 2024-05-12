import logging
from collections import OrderedDict

import torch
import torch.nn as nn

import torch.optim as optim
import random
import math
import numpy as np
import model.lr_scheduler as lr_scheduler
import torch.optim.lr_scheduler as lrs
from model.basicsr.metrics import calculate_niqe
import model.calculate_lpips as calculate_lpips

import os
import model.networks as networks
import model.ppo as ppo
from .base_model import BaseModel

logger = logging.getLogger('base')
import numpy as np
import model.metrics as Metrics
import cv2


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)

        self.opt = opt

        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))

        # ----------------- define PPO -------------------
        self.ppo = ppo.PPO(3, 27, 1536, True)
        self.memory = ppo.Memory()
        #-------------------------------------------------

        self.schedule_phase = None
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            wd_G = opt['weight_decay_G'] if opt['weight_decay_G'] else 0
            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"],
                                         weight_decay=wd_G,
                                         betas=(opt['train']["optimizer"]['beta1'], opt['train']["optimizer"]['beta2']))
            self.optimizers.append(self.optG)

            # schedulers
            if opt['train']["optimizer"]['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, opt['train']["optimizer"]['lr_steps'],
                                                         restarts=opt['train']["optimizer"]['restarts'],
                                                         weights=opt['train']["optimizer"]['restart_weights'],
                                                         gamma=opt['train']["optimizer"]['lr_gamma'],
                                                         clear_state=opt['train']["optimizer"]['clear_state'],
                                                         lr_steps_invese=opt.get('lr_steps_inverse', [])))

            elif opt['train']["optimizer"]['lr_scheme'] == 'step':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lrs.StepLR(
                            optimizer,
                            step_size=opt['train']["optimizer"]['lr_decay'],
                            gamma=opt['train']["optimizer"]['lr_gamma']
                        ))

            elif opt['train']["optimizer"]['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, opt['train']["optimizer"]['T_period'],
                            eta_min=opt['train']["optimizer"]['eta_min'],
                            restarts=opt['train']["optimizer"]['restarts'],
                            weights=opt['train']["optimizer"]['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()
        self.data_ok = 0

    def feed_data(self, data):
        self.data = self.set_device(data)


    def optimize_parameters(self):
        self.schedulers[0].step()

        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def policy_optimize_parameters_PPO(self, iter):
        argsT = 10
        batch_time = AverageMeter()
        reward_list = [AverageMeter() for _ in range(argsT)]
        final_reward = 0

        self.netG.eval()
        ep_r = 0
        b, c, h, w = self.data['HR'].shape
        shape = self.data['SR'].shape
        img = torch.randn(shape).cuda()

        # CSA: Coarse diffusion
        with torch.no_grad():
            self.set_new_noise_schedule(
                self.opt['model']['beta_schedule']['val'], n_timestep=10, schedule_phase='val')
            _, img10, _, results = self.netG.super_resolution_noise_in(self.data['SR'], img,
                                                                       continous=False)

        HR = self.data['HR'].detach().float().cpu()
        HR = [Metrics.tensor2img(HR[i, :, :, :]) for i in range(b)]

        # the initial state for FSR
        state = results[-2]
        img = results[-2]
        count = 0
        action_lists = []
        for step in range(0, argsT):
            # according to the state, choose an action
            if step == 0:
                action = self.ppo.select_action(state.to(0), self.memory, restart_batch=True)
            else:
                action = self.ppo.select_action(state.to(0), self.memory)
            if step == argsT - 1:
                t = 0
                with torch.no_grad():
                    self.set_new_noise_schedule(
                        self.opt['model']['beta_schedule']['val'], n_timestep=100, schedule_phase='val')
                    self.SR = self.netG.super_resolution_train(self.data['SR'], t,
                                                               img, continous=False)
                srnew_np = self.SR.detach().float().cpu()
                srnew_np = [Metrics.tensor2img(srnew_np[i, :, :, :]) for i in range(b)]
                niqe_i = [calculate_niqe.test_single(HR[i], srnew_np[i]) for i in range(b)]
                reward = [1.0 / niqe_i[i] for i in range(b)]
                ep_r += sum(reward)
                final_reward = reward  
                action_lists.append(1)
            else:
                new_img = []
                reward = []
                for i in range(b):
                    if action[i].item() == 1:
                        count += 1
                        self.SR = results[-2][i:i + 1, ...]
                        r = 0.0
                    else:
                        with torch.no_grad():
                            self.set_new_noise_schedule(
                                self.opt['model']['beta_schedule']['val'], n_timestep=100, schedule_phase='val')
                            t = 9 - step
                            self.SR = self.netG.super_resolution_train(self.data['SR'][i:i + 1, ...], t,
                                                                       img[i:i + 1, ...], continous=False)
                        r = 0.000

                    reward.append(r)
                    new_img.append(self.SR)
                    ep_r += r

                state = torch.cat(new_img, 0)
                img = state

            rewards = torch.from_numpy(np.array(reward)).float()
            reward_list[step - 1].update(rewards.data.mean(), self.data['HR'].size(0))
            self.memory.rewards.append(rewards)

        l_pix = self.ppo.update(self.memory)
        self.memory.clear_memory()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['reward'] = ep_r  # .item()
        self.log_dict['remove'] = count  # .item()

        if iter % 1e4 == 0:
            save_path = './ppo_models/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.ppo.policy, save_path + 'ppo_model_' + str(iter) + '.pth')

        if iter % 200 == 0:
            print('iter: ', iter, 'final_reward: ', final_reward, 'remove state number: ', count, self.data_ok)

    def test_PPO(self, continous=False):
        argsT = 10
        batch_time = AverageMeter()
        reward_list = [AverageMeter() for _ in range(argsT)]
        self.netG.eval()

        self.ppo.policy.eval()

        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                b, c, h, w = self.data['HR'].shape
                shape = self.data['SR'].shape
                img = torch.randn(shape).cuda()  # random noise

                HR = self.data['HR'].detach().float().cpu()
                HR = Metrics.tensor2img(HR)

                with torch.no_grad():
                    self.set_new_noise_schedule(
                        self.opt['model']['beta_schedule']['val'], n_timestep=10, schedule_phase='val')

                    _, img10, _, results = self.netG.super_resolution_noise_in(self.data['SR'], img,
                                                                               continous=False)

                    self.set_new_noise_schedule(
                        self.opt['model']['beta_schedule']['val'], n_timestep=100, schedule_phase='val')

                state = results[-2] #img
                img = results[-2]

                count = 0
                count_step = []
                for step in range(0, argsT):
                    if step == 0:
                        action = self.ppo.select_action(state.to(0), self.memory, restart_batch=True,
                                                        training=False)
                    else:
                        action = self.ppo.select_action(state.to(0), self.memory, training=False)

                    if step == argsT - 1:
                        t = 0
                        with torch.no_grad():
                            self.set_new_noise_schedule(
                                self.opt['model']['beta_schedule']['val'], n_timestep=100, schedule_phase='val')
                            img = self.netG.super_resolution_train(self.data['SR'], t,
                                                                   img, continous=False)
                    else:
                        a = torch.argmax(action).item()
                        print('the current action is: ', a)
                        if a == 1:
                            count += 1
                            count_step.append(argsT - (step + 1))
                            img = results[-2]
                        else:
                            with torch.no_grad():
                                self.set_new_noise_schedule(
                                    self.opt['model']['beta_schedule']['val'], n_timestep=100, schedule_phase='val')
                                t = argsT - 1 - step
                                img = self.netG.super_resolution_train(self.data['SR'], t, img, continous=False)

                    state = img

                # ----------------------- Metric evaluation -------------------------------
                self.SR = img
                srnew_np = Metrics.tensor2img(self.SR)

                # calculate PSNR
                psnr = Metrics.calc_psnr_ycbcr(srnew_np, HR)

                # calculate SSIM
                ssim = Metrics.calculate_ssim(srnew_np, HR)

                # calculate NIQE
                niqe = calculate_niqe(srnew_np, 0, input_order='HWC', convert_to='y')

                # calculate LPIPS
                lpips = calculate_lpips.test_single(HR, srnew_np)

                print('image name: {}, reduce: {}, steps: {}, psnr:{}, ssim:{}, niqe:{}, lpips:{}'.format(
                    self.data['HR_path'], count, count_step, psnr, ssim, niqe, lpips))

                fileName = './FFHQ_CelebAHQ_PPO.txt'
                with open(fileName, 'a+') as file:
                    file.write('image name: {}, reduce: {}, steps: {}, psnr:{}, ssim:{}, niqe:{}, lpips:{}'.format(
                        self.data['HR_path'], count, count_step, psnr, ssim, niqe, lpips)+'\n')

        return psnr, ssim, niqe, lpips,  self.data['HR_path'][2:-2]


    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, n_timestep, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = None
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, n_timestep, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, n_timestep, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach()[0].float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()  # [0]
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        resume = self.opt['path']['resume']
        if resume == 0:
            if load_path != '.':
                logger.info(
                    'Loading model from {}'.format(load_path))
                gen_path = '{}_gen.pth'.format(load_path)
                # gen
                network = self.netG
                if isinstance(self.netG, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(
                    gen_path), strict=False)
        else:
            if load_path is not None:
                logger.info(
                    'Loading pretrained model for G [{:s}] ...'.format(load_path))
                gen_path = '{}_gen.pth'.format(load_path)
                opt_path = '{}_opt.pth'.format(load_path)
                # gen
                network = self.netG
                if isinstance(self.netG, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(
                    gen_path), strict=(not self.opt['model']['finetune_norm']))
                if self.opt['phase'] == 'train':
                    # optimizer
                    opt = torch.load(opt_path)
                    self.optG.load_state_dict(opt['optimizer'])
                    self.begin_step = opt['iter']
                    self.begin_epoch = opt['epoch']


