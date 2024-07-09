from torch import nn
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import numpy as np
import math

class DiffussionModel(nn.Module):
    def __init__(self, model, beta_s=0.0001, beta_e=0.02, timestep=1000, mode='train', device='cpu'):
        super().__init__()
        self.model = model
        self.beta_s = beta_s
        self.beta_e = beta_e
        self.timestep = timestep
        self.mode = mode
        self.device = device

        # need to build alpha beta tensor , size: timespace + 1 => 0 ~ 1000
        self.beta_tensor = self.build_beta(beta_s, beta_e, timestep)
        self.alpha_tensor = 1 - self.beta_tensor
        # self.alpha_hat_tensor = F.pad(torch.cumprod(self.alpha_tensor, dim=0)[:-1], (1,0), value=1.0)
        self.alpha_hat_tensor = torch.cumprod(self.alpha_tensor, dim=0)

        # here for generate
        self.alpha_hat_tensor_prev = F.pad(self.alpha_hat_tensor[:-1], (1,0), value=1.0)  # alpha_{t-1}
        self.sigma = torch.sqrt(((1 - self.alpha_hat_tensor_prev)*self.beta_tensor) / (1 - self.alpha_hat_tensor))
        
        
    
    def build_beta(self, beta_s, beta_e, timestep):
        return torch.linspace(beta_s, beta_e, timestep, dtype=torch.float64)
    
    def mul(self, a, x, t):
        shape = x.size()
        for i in range(shape[0]):
            tmp = t[i]
            x[i] = a[tmp] * x[i]
        
        return x
    
    def generate(self, batch_size, img_size):
        x_last = torch.randn((batch_size, 3, img_size, img_size), device=self.device)
        
        for i in tqdm(range(self.timestep-1, -1, -1)):
            t = torch.tensor([i]*batch_size, device=self.device)
            
            z = torch.randn_like(x_last, device=self.device)
            # front_model = self.mul((1-self.alpha_tensor) / (torch.sqrt(1-self.alpha_hat_tensor)), self.model(x_last, t), t)
            # first = self.mul((1/torch.sqrt(self.alpha_tensor)), x_last - front_model, t)
            # second = self.mul(torch.sqrt(((1-self.alpha_hat_tensor_prev)*self.beta_tensor) / (1 - self.alpha_hat_tensor)), z, t)
            # front_model = ((1-self.alpha_tensor[i])/(1-self.alpha_hat_tensor[i])**0.5) * self.model(x_last, t)
            front_model = ((self.beta_tensor[i])/(1-self.alpha_hat_tensor[i])**0.5) * self.model(x_last, t)
            first = (1/self.alpha_tensor[i]**0.5) * (x_last - front_model)
            second = ((self.beta_tensor[i] * (1-self.alpha_hat_tensor_prev[i]) / (1-self.alpha_hat_tensor[i]))**0.5)*z
            
            x_last = first + second
        
        return x_last

    def DDIM_generate(self, noise, eta):
        batch_size = noise.size()[0]
        steps = [i for i in range(999, -1, -20)]
        steps.append(0)
        x_last = noise.clone()
        for i in tqdm(range(len(steps)-1)):
            step = steps[i]
            last_step = steps[i+1]

            alpha_t = self.alpha_hat_tensor[step]
            alpha_t_Minus1 = self.alpha_hat_tensor[last_step] if last_step > 0 else 0.9999
            #print(i, steps[i], steps[i-1], alpha_t, alpha_t_Minus1)
            sigma = eta * np.sqrt((1-alpha_t_Minus1) / (1-alpha_t)) * np.sqrt(1-alpha_t/alpha_t_Minus1)

            
            
            t = torch.tensor([steps[i]]*batch_size, device=self.device)
            z = torch.randn_like(x_last, device=self.device)

            first = (np.sqrt(alpha_t_Minus1)/np.sqrt(alpha_t))*(x_last - np.sqrt(1-alpha_t)*self.model(x_last, t))
            second = np.sqrt(1 - alpha_t_Minus1 - sigma**2)*self.model(x_last, t)
            third = sigma * z

            x_last = first + second + third
        
        return x_last
            




    def forward(self, x, model_mode='DDPM'):
        if self.mode == 'train':
            shape = x.size()
            # x: a "batch" of data
            # sample a t -> size: (batch_size, )
            
            t = torch.randint(0, self.timestep, (shape[0], ), device=self.device).long()
            # sample a noise (epsilon)
            noise = torch.randn_like(x, device=self.device)
            #print(noise)
            sq_alpha_x0 = self.mul(torch.sqrt(self.alpha_hat_tensor), x.clone(), t)
            sq_one_alpha_epsilon = self.mul(torch.sqrt(1 - self.alpha_hat_tensor), noise.clone(), t)
            #print(noise)
            #os._exit(0)
            tmp_out = self.model(sq_alpha_x0+sq_one_alpha_epsilon, t)

            return nn.MSELoss()(noise, tmp_out)
        
        else:
            if model_mode == 'DDPM':
                return self.generate(*x)
            else:
                return self.DDIM_generate(*x)

class ConDiffussionModel(nn.Module):
    def __init__(self, model, beta_s=0.0001, beta_e=0.02, timestep=1000, mode='train', device='cpu', class_e=None):
        super().__init__()
        self.model = model
        self.beta_s = beta_s
        self.beta_e = beta_e
        self.timestep = timestep
        self.mode = mode
        self.device = device
        self.class_e = class_e

        # need to build alpha beta tensor , size: timespace + 1 => 0 ~ 1000
        self.beta_tensor = self.build_beta(beta_s, beta_e, timestep)
        #self.beta_tensor = self.cosine_beta_schedule(1000)
        self.alpha_tensor = 1 - self.beta_tensor
        self.alpha_hat_tensor = F.pad(torch.cumprod(self.alpha_tensor, dim=0)[:-1], (1,0), value=1.0)
        self.alpha_hat_tensor = torch.cumprod(self.alpha_tensor, dim=0)

        # here for generate
        self.alpha_hat_tensor_prev = F.pad(self.alpha_hat_tensor[:-1], (1,0), value=1.0)  # alpha_{t-1}
        self.sigma = torch.sqrt(((1 - self.alpha_hat_tensor_prev)*self.beta_tensor) / (1 - self.alpha_hat_tensor))
        
        
    def cosine_beta_schedule(self, timesteps, s=0.008, **kwargs):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """

        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def build_beta(self, beta_s, beta_e, timestep):
        return torch.linspace(beta_s, beta_e, timestep, dtype=torch.float64)
    
    def mul(self, a, x, t):
        shape = x.size()
        for i in range(shape[0]):
            tmp = t[i]
            x[i] = a[tmp] * x[i]
        
        return x
    
    def generate(self, batch_size, img_size, label, report=False):
        x_last = torch.randn((batch_size, 3, img_size, img_size), device=self.device)
        report_img = None
        if report :
            report_img = x_last[0].clone().unsqueeze(dim=0)
            
        c = label
        
        for i in tqdm(range(self.timestep-1, -1, -1)):
            t = torch.tensor([i]*batch_size, device=self.device)
            # c = torch.tensor([label]*batch_size, device=self.device)
            
            z = torch.randn_like(x_last, device=self.device)
            # front_model = self.mul((1-self.alpha_tensor) / (torch.sqrt(1-self.alpha_hat_tensor)), self.model(x_last, t), t)
            # first = self.mul((1/torch.sqrt(self.alpha_tensor)), x_last - front_model, t)
            # second = self.mul(torch.sqrt(((1-self.alpha_hat_tensor_prev)*self.beta_tensor) / (1 - self.alpha_hat_tensor)), z, t)
            # front_model = ((1-self.alpha_tensor[i])/(1-self.alpha_hat_tensor[i])**0.5) * self.model(x_last, t)
            front_model = ((self.beta_tensor[i])/(1-self.alpha_hat_tensor[i])**0.5) * self.model(x_last, t, c)
            first = (1/self.alpha_tensor[i]**0.5) * (x_last - front_model)
            second = ((self.beta_tensor[i] * (1-self.alpha_hat_tensor_prev[i]) / (1-self.alpha_hat_tensor[i]))**0.5)*z
            
            x_last = first + second

        return x_last

        #     if i == 799 or i == 599 or i == 399 or i == 199:
        #         if report:
        #             report_img = torch.concatenate((report_img, x_last[0].clone().unsqueeze(dim=0)), dim=0)
        # if report:
        #     report_img = torch.concatenate((report_img, x_last[0].clone().unsqueeze(dim=0)), dim=0)
        #     return x_last, report_img
        # else: 
        #     return x_last

    def DDIM_generate(self, noise, eta):
        batch_size = noise.size()[0]
        steps = [i for i in range(999, -1, -2)]
        steps.append(0)
        x_last = noise.clone()
        for i in tqdm(range(len(steps)-1)):
            step = steps[i]
            last_step = steps[i+1]

            alpha_t = self.alpha_hat_tensor[step]
            alpha_t_Minus1 = self.alpha_hat_tensor[last_step] if last_step > 0 else 0.9999
            #print(i, steps[i], steps[i-1], alpha_t, alpha_t_Minus1)
            sigma = eta * np.sqrt((1-alpha_t_Minus1) / (1-alpha_t)) * np.sqrt(1-alpha_t/alpha_t_Minus1)

            
            
            t = torch.tensor([steps[i]]*batch_size, device=self.device)
            z = torch.randn_like(x_last, device=self.device)

            first = (np.sqrt(alpha_t_Minus1)/np.sqrt(alpha_t))*(x_last - np.sqrt(1-alpha_t)*self.model(x_last, t))
            second = np.sqrt(1 - alpha_t_Minus1 - sigma**2)*self.model(x_last, t)
            third = sigma * z

            x_last = first + second + third
        
        return x_last
            




    def forward(self, x, label=None, report=False,  model_mode='DDPM'):
        if label is None:
            label = torch.tensor([1]*500)
        if self.mode == 'train':
            shape = x.size()
            # x: a "batch" of data
            # sample a t -> size: (batch_size, )
            
            t = torch.randint(0, self.timestep, (shape[0], ), device=self.device).long()
            # sample a noise (epsilon)
            noise = torch.randn_like(x, device=self.device)
            #print(noise)
            sq_alpha_x0 = self.mul(torch.sqrt(self.alpha_hat_tensor), x.clone(), t)
            sq_one_alpha_epsilon = self.mul(torch.sqrt(1 - self.alpha_hat_tensor), noise.clone(), t)
            #print(noise)
            #os._exit(0)
            tmp_out = self.model(sq_alpha_x0+sq_one_alpha_epsilon, t, label)

            return nn.MSELoss()(noise, tmp_out)
        
        else:
            if model_mode == 'DDPM':
                return self.generate(*x, label, report=report)
            else:
                return self.DDIM_generate(*x)
