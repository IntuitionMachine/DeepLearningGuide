'''VGG16 neural network with Shapley Additive Explanations framework from p. 217 of the book.

Based on https://github.com/slundberg/shap'''

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import skimage, shap, numpy as np
import requests
from skimage.segmentation import slic
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as pl

model = VGG16()
r = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
feature_names = r.json()
file = "data/image_1.jpg"
img = image.load_img(file, target_size=(224, 224))
img_orig = image.img_to_array(img)
segments_slic = slic(img, n_segments=50, compactness=30, sigma=3)

def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background
    return out

def f(z):
    return model.predict(preprocess_input(mask_image(z, segments_slic, img_orig, 255)))

def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out


def main():
    explainer = shap.KernelExplainer(f, np.zeros((1,50)))
    shap_values = explainer.shap_values(np.ones((1,50)), nsamples=1000)
    preds = model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))
    top_preds = np.argsort(-preds)

    colors = []
    for l in np.linspace(1,0,100):
        colors.append((245/255,39/255,87/255,l))
    for l in np.linspace(0,1,100):
        colors.append((24/255,196/255,93/255,l))
    cm = LinearSegmentedColormap.from_list("shap", colors)

    fig, axes = pl.subplots(nrows=1, ncols=4, figsize=(12,4))
    inds = top_preds[0]
    axes[0].imshow(img)
    axes[0].axis('off')
    max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
    for i in range(3):
        m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
        axes[i+1].set_title(feature_names[str(inds[i])][1])
        axes[i+1].imshow(img.convert('LA'), alpha=0.15)
        im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
        axes[i+1].axis('off')
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
    cb.outline.set_visible(False)
    pl.show()


main()
