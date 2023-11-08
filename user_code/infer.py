import numpy as np

def load_model():
    # you can load your model here and return it
    return None

def infer(model, imgs):
    # you can do your inference here and return the result
    
    result = np.array([np.array([np.random.rand(6) for i in range(100)]) for j in range(imgs.shape[0])])
    return result