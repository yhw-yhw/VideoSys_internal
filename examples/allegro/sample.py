from videosys import AllegroConfig,AllegroPipeline
from videosys import VideoSysEngine

def run_base():
    config = AllegroConfig("/data4/ryd_workspace/models/Allegro",num_gpus=1)
    engine = VideoSysEngine(config)
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
    video = engine.generate(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        num_inference_steps=20,
        num_frames=88,
        seed=42,
        max_sequence_length=512
    ).video[0]
    engine.save_video(video, f"./outputs/test.mp4")
    
    
def run_pab():
    config = AllegroConfig("/data4/ryd_workspace/models/Allegro",num_gpus=1,enable_pab=True)
    engine = VideoSysEngine(config)
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
    video = engine.generate(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        # height = 480,
        # width = 720,
        guidance_scale=7.5,
        num_inference_steps=20,
        num_frames=88,
        seed=42,
        max_sequence_length=512
    ).video[0]
    engine.save_video(video, f"./outputs/test_with_pab.mp4")

if __name__ == "__main__":
    run_base()
    # run_pab()