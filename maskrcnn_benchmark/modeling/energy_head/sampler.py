import torch
import torch.nn.functional as F
from maskrcnn_benchmark.modeling import registry

@registry.SAMPLER.register("SGLD")
class SGLD(object):
    '''Class for Stochastic Gradient Langevin Dynamics'''

    def __init__(self, cfg):

        self.sgld_lr = float(cfg.SAMPLER.LR)
        self.sgld_var = float(cfg.SAMPLER.VAR)
        self.grad_clip = float(cfg.SAMPLER.GRAD_CLIP)
        self.iters = cfg.SAMPLER.ITERS
    def sample(self, model, im_graph, scene_graph, bbox, mode, joint=False):

        model.train()
        if not joint:
            scene_graph.requires_grad(mode) #For autograd
        
        if mode == 'predcls':
            noise = torch.rand_like(scene_graph.edge_states)

            for _ in range(self.iters):
                noise.normal_(0, self.sgld_var)
                scene_graph.edge_states.data.add_(noise.data)

                edge_states_grads = torch.autograd.grad(model(im_graph, scene_graph, bbox).sum(), [scene_graph.edge_states], retain_graph=True)[0]
                edge_states_grads.data.clamp_(-self.grad_clip, self.grad_clip)
                
                scene_graph.edge_states.data.add_(edge_states_grads, alpha=-self.sgld_lr)

                scene_graph.edge_states.detach()
                # scene_graph.edge_states = F.softmax(scene_graph.edge_states, dim=1)
                # scene_graph.edge_states = scene_graph.edge_states/torch.sum(scene_graph.edge_states, dim=1,  keepdim=True)
        else:
            noise = torch.rand_like(scene_graph.edge_states)
            noise2 = torch_rand_like(scene_graph.node_states)

            for _ in range(self.iters):
                noise.normal_(0, self.sgld_var)
                noise2.normal_(0, self.sgld_var)

                scene_graph.edge_states.data.add_(noise.data)
                scene_graph.node_states.data.add_(noise2.data)

                edge_states_grads, node_states_grads = torch.autograd.grad(model(im_graph, scene_graph, bbox).sum(), [scene_graph.edge_states, scene_graph.node_states], retain_graph=True)
                edge_states_grads.data.clamp_(-self.grad_clip, self.grad_clip)
                node_states_grads.data.clamp_(-self.grad_clip, self.grad_clip)

                scene_graph.edge_states.data.add_(edge_states_grads, alpha=-self.sgld_lr)
                scene_graph,node_states.data.add_(node_states_grads, alpha=-self.sgld_lr)

                # scene_graph.edge_states = scene_graph.edge_states/torch.sum(scene_graph.edge_states, dim=1,  keepdim=True)
                # scene_graph.node_states = scene_graph.node_states/torch.sum(scene_graph.node_states, dim=1,  keepdim=True)

                scene_graph.edge_states.detach()
                scene_graph.node_states.detach()

        return scene_graph

def build_sampler(cfg):

    sampler = registry.SAMPLER[cfg.SAMPLER.NAME]
    return sampler(cfg)