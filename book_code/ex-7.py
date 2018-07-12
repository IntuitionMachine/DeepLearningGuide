'''VGG16 neural network with Shapley Additive Explanations framework from p. 217 of the book.

Based on https://github.com/slundberg/shap'''

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import skimage

'''Defines an image mask.'''
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

'Runs a prediction on the preprocessing masked image'
def f(z):
    return model.predict(preprocess_input(mask_image(z, segments_slic, img_orig, 255)))

'Runs p. 217 Shapley code.'
def main():
    model = VGG16()
    file = "data/apple_strawberry.jpg"
    img = image.load_img(file, target_size=(224, 224))
    img_orig = image.img_to_array(img)
    segments_slice = skimage.segmentation.slic(img, n_segments=50, compactness=30, sigma=3)
    explainer = shap.KernelExplainer(f, np.zeros((1,50)))
    shap_values = explainer.shap_values(np.ones((1,50)), nsamples=1000)
    preds = model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))
    top_preds = np.argsort(-preds)

main()
