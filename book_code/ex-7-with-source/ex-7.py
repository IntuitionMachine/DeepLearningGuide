'''VGG16 neural network with Shapley Additive Explanations framework from p. 217 of the book.

Based on https://github.com/slundberg/shap'''

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import skimage, shap, numpy as np
from skimage.segmentation import slic

model = VGG16()
file = "data/image_1.jpg"
img = image.load_img(file, target_size=(224, 224))
img_orig = image.img_to_array(img)
#segments_slice = skimage.segmentation.slic(img, n_segments=50, compactness=30, sigma=3)
segments_slice = slic(img, n_segments=50, compactness=30, sigma=3)

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
    return model.predict(preprocess_input(mask_image(z, segments_slice, img_orig, 255)))

def main():
    explainer = shap.KernelExplainer(f, np.zeros((1,50)))
    shap_values = explainer.shap_values(np.ones((1,50)), nsamples=1000)
    preds = model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))
    top_preds = np.argsort(-preds)
    #shap.image_plot(shape_values, preds)

main()
