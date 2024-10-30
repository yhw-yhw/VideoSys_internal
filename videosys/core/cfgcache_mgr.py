from videosys.utils.logging import logger
import torch.fft

@torch.no_grad()
def fft(tensor):
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    B, C, H, W = tensor.size()
    radius = min(H, W) // 5
            
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_x, center_y = W // 2, H // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2

    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask
            
    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft

class CFGCacheConfig():
    def __init__(
        self,
        threshold_l:int=0,
        threshold_r:int=1000    
    ) -> None:
        self.threshold_l = threshold_l
        self.threshold_r = threshold_r

class CFGCacheManager():
    def __init__(
        self,
        config:CFGCacheConfig
    ):
        self.config = config
    
    def if_use_cfgcache(self,cur_timestep)->bool:
        if cur_timestep>self.config.threshold_l and cur_timestep<self.config.threshold_r:
            return True
        return False
def set_cfgcache_manager(config:CFGCacheConfig):
    global CFGCACHE_MANAGER
    CFGCACHE_MANAGER = CFGCacheManager(config)
    
def if_use_cfgcache(cur_timestep)->bool:
    return CFGCACHE_MANAGER.if_use_cfgcache(cur_timestep)