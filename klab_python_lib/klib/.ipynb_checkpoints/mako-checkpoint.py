from .imports import *

# from .expfile import *
# from .analysis import *
# from .mathutil import *
# from .plotutil import *
# from .imagutil import *
# from .mako import *
# from .adam import *

def loadBMPs(path, nameFormat):
    names = arr(os.listdir(path))
    print(names)
    inds = arr([int((i.replace(nameFormat,"")).replace(".bmp","")) for i in names])
    sort = np.argsort(inds)
    names = names[sort]
    imgs = []
    for name in names:
        img  = arr(Image.open(path+name))
        imgs.append(img)
    
    return imgs

def sortImgs(exp, imgs):
    """Sort list of imported images (imgs) by hdf5 experiment object (exp) returned from Chimera."""
    key = exp.key
    reps = exp.reps
    sort = np.argsort(key)
    shape = imgs.shape
    imgs = np.reshape(imgs, (len(key), reps, shape[-2], shape[-1]))
    return key[sort], imgs[sort]