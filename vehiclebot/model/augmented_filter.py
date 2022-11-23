
from .filter import *

def augmentation(img, num_iters : int = 50, aug_max_combinations : int = 2) -> list:
    imgs = []
    #Generate augmentations randomly
    always = [filter__unsharp_mask, filter__removeShadow]
    methods = [filter__threshold, filter__adaptiveThreshold, filter__linear_brightness, filter__gamma_correction]
    for _ in range(num_iters):
        num_tasks = random.randint(0, aug_max_combinations)
        for t in always:
            img = t(img)
        for _ in range(num_tasks):
            img = random.choice(methods)(img)
        imgs.append(img)
        
    return imgs
