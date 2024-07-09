from p1_model import ConDiffussionModel
from MyUNet import UNet
#from UNet import UNet
from torchview import draw_graph
import torch
import graphviz
graphviz.set_jupyter_format('png')

embedding_noise = torch.empty((0, 1, 32, 32))
for i in range(10):
    embedding_noise = torch.concatenate((embedding_noise, torch.randn((1,1,32,32))), dim=0)
unet = UNet(class_embeded=embedding_noise.to('cuda'), device='cuda').to('cuda')
DDPM = ConDiffussionModel(unet, device='cuda').to('cuda')
#unet = UNet()
DDPM.eval()
batch_size = 2
model_graph = draw_graph(unet, input_data=[torch.randn((batch_size, 3, 32, 32)).to('cuda'), torch.tensor([0]*2).to('cuda'), torch.tensor([0]*2).to('cuda')], 
                         device='cuda', save_graph=True, filename="p1IMAGE_real_5", depth=1)  # , hide_inner_tensors=False, hide_module_functions=False
model_graph.visual_graph