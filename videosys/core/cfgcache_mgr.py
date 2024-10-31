from videosys.utils.logging import logger
import torch.fft

CFGCACHE_MANAGER=None

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
        threshold_r:int=1000,
        threshold_range:int=6    
    ) -> None:
        self.threshold_l = threshold_l
        self.threshold_r = threshold_r
        self.threshold_range = threshold_range
        
class CFGCacheManager():
    def __init__(
        self,
        config:CFGCacheConfig,
        step_counter:int=0
    ):
        self.config = config
        self.step_counter = step_counter
        logger.info(f"CFGCACHE: threshold: l: {config.threshold_l},r: {config.threshold_r} threshold_range: {config.threshold_range}")
    
    def if_use_cfgcache(self,cur_timestep)->bool:
        cur_timestep = int(cur_timestep)
        if cur_timestep>self.config.threshold_l and cur_timestep<self.config.threshold_r:
            return True
        return False
def set_cfgcache_manager(config:CFGCacheConfig):
    global CFGCACHE_MANAGER
    CFGCACHE_MANAGER = CFGCacheManager(config)
    
def if_use_cfgcache(cur_timestep)->bool:
    return CFGCACHE_MANAGER.if_use_cfgcache(cur_timestep)

def enable_cfgcache()->bool:
    if CFGCACHE_MANAGER==None:
        return False
    return True

def cfgcache_step():
    ### there is a bug, it seems like step counter is not increasing TODO
    CFGCACHE_MANAGER.step_counter = CFGCACHE_MANAGER.step_counter+1

def if_get_accelerated():
    if CFGCACHE_MANAGER.step_counter % CFGCACHE_MANAGER.config.threshold_range==0 and CFGCACHE_MANAGER.step_counter!=0:
        return True
    return False
