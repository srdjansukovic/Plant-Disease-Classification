from tensorflow.keras.models import Model
import tensorflow as tf
from plant_classes import plant_classes
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import imutils
import cv2
from prepare_images import prepare_heatmap


class GradCAM:

    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_HOT):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(src=heatmap, colormap=colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return heatmap, output


def apply_gradcam(classifier):
    heatmap_paths = prepare_heatmap('Dataset-mini\\valid')
    for img_path in heatmap_paths:
        orig = cv2.imread(img_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(orig, (128, 128))
        # load the input image from disk (in Keras/TensorFlow format) and
        # preprocess it
        image = load_img(img_path, target_size=(128, 128))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # image = imagenet_utils.preprocess_input(image)

        # use the network to make predictions on the input image and find
        # the class label index with the largest corresponding probability
        preds = classifier.predict(image)
        i = np.argmax(preds[0])
        print(preds, i)
        # decode the ImageNet predictions to obtain the human-readable label
        # decoded = imagenet_utils.decode_predictions(preds)
        # (imagenetID, label, prob) = decoded[0][0]
        # label = "{}: {:.2f}%".format(label, prob * 100)
        label = f'{plant_classes[i]}, {round(preds[0][i] * 100, 3)}%'
        print("[INFO] {}".format(label))

        # initialize our gradient class activation map and build the heatmap
        cam = GradCAM(classifier, i)
        heatmap = cam.compute_heatmap(image)
        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the image
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

        # draw the predicted label on the output image
        # cv2.rectangle(output, (0, 0), (540, 20), (255, 255, 255), -1)
        # cv2.putText(output, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # display the original image and resulting heatmap and output image
        # to our screen
        output = np.hstack([orig, heatmap, output])
        blank_image = np.zeros((40, 900, 3), np.uint8)
        cv2.rectangle(blank_image, (0, 0), (900, 40), (255, 255, 255), -1)
        cv2.putText(blank_image, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        output = imutils.resize(output, width=900)
        output = np.vstack([output, blank_image])
        cv2.imshow("Output", output)
        cv2.waitKey(0)
