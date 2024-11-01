from videosys import AllegroConfig,AllegroPipeline
from videosys import VideoSysEngine
import torch
import time
def caculate_time(method,*args):
    start = time.time()
    method(*args)
    return time.time()-start

def run_base(positive_prompt,negative_prompt):
    config = AllegroConfig("/data4/ryd_workspace/models/Allegro",num_gpus=1)
    engine = VideoSysEngine(config)
    with torch.no_grad():
        video = engine.generate(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=20,
            num_frames=88,
            seed=42,
            max_sequence_length=512
        ).video[0]
    engine.save_video(video, f"./outputs/test_base.mp4")

def run_pab(positive_prompt,negative_prompt):
    config = AllegroConfig("/data4/ryd_workspace/models/Allegro",num_gpus=1,enable_pab=True)
    engine = VideoSysEngine(config)
    video = engine.generate(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        num_inference_steps=20,
        num_frames=88,
        seed=42,
        max_sequence_length=512
    ).video[0]
    engine.save_video(video, f"./outputs/test_pab.mp4")

def run_cfgcache(positive_prompt,negative_prompt):
    config = AllegroConfig("/data4/ryd_workspace/models/Allegro",num_gpus=1,enable_cfgcache=True)
    engine = VideoSysEngine(config)
    video = engine.generate(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        num_inference_steps=20,
        num_frames=88,
        seed=42,
        max_sequence_length=512
    ).video[0]
    engine.save_video(video, f"./outputs/test_cfgcache.mp4")
    
def run_cfgcache_pab(positive_prompt,negative_prompt):
    config = AllegroConfig("/data4/ryd_workspace/models/Allegro",num_gpus=1,enable_cfgcache=True,enable_pab=True)
    engine = VideoSysEngine(config)
    video = engine.generate(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        num_inference_steps=20,
        num_frames=88,
        seed=42,
        max_sequence_length=512
    ).video[0]
    engine.save_video(video, f"./outputs/test_cfgcache_pab.mp4")
    
if __name__ == "__main__":
    # user_prompt = "A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this location might be a popular spot for docking fishing boats."
    user_prompt = "Sunset over the sea."
    positive_prompt = """
    (masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
    {} 
    emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
    sharp focus, high budget, cinemascope, moody, epic, gorgeous
    """
    positive_prompt=positive_prompt.format(user_prompt.lower().strip())
    print(positive_prompt)
    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """
    execute_time = {
        #only choose one method! The Global variations may cause bugs.
    "base_time" : caculate_time(run_base,positive_prompt,negative_prompt), #645.3020467758179
    # "cfgcache_time" : caculate_time(run_cfgcache,positive_prompt,negative_prompt), #579.5817785263062
    # "pab_time" : caculate_time(run_pab,positive_prompt,negative_prompt), #558.1389479637146
    # "cfgcache_pab_time" : caculate_time(run_cfgcache_pab,positive_prompt,negative_prompt) #496.5060832500458
    }
    
    print(execute_time)
    