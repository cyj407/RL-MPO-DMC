
from dm_control import suite
from dm_control.suite.wrappers import pixels
from moviepy.editor import ImageSequenceClip
import numpy as np
import dm_control2gym
from my_mpo.actor import Actor
import torch
from moviepy.editor import ImageSequenceClip

def main():
    
    # pretrained actor path
    load_path = 'hopper/model/model_700.pt'

    # set up the rendering window
    dm_control2gym.create_render_mode('camera0', show=True, return_pixel=True, 
                                        height=480, width=640, camera_id=0)

    env = dm_control2gym.make(domain_name='hopper', task_name='stand')
    env.seed(666)

    ## EXAMPLE from dm_control2gym
    # env.reset()
    # for t in range(1000):
    #     observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    #     env.render()
    
    actor = Actor(env)
    checkpoint = torch.load(load_path)
    actor.load_state_dict(checkpoint['actor_state_dict'])

    videos = []
    state = env.reset()
    for t in range(300):
        action = actor.action( torch.from_numpy(state).type(torch.float32)).cpu().numpy()
        state, reward, done, _ = env.step(action)
        frame = env.render(mode='camera0')
        videos.append(frame)
        if done:
            break
    
    clip = ImageSequenceClip(videos, fps=50)
    clip.write_gif("hopper.gif")

if __name__ == '__main__':
    main()