import tensorflow as tf
from tensorflow.keras.models import Model
from .blazepose_layers import BlazeBlock
from tensorflow.keras import layers

class BlazePose():
    def __init__(self, num_keypoints: int):

        self.num_keypoints = num_keypoints

        self.conv1 = tf.keras.layers.Conv2D(
            filters=24, kernel_size=3, strides=(2, 2), padding='same', activation='relu'
        )
        print(self.conv1.get_weights())

        self.conv2_1 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)
        ])

        self.conv2_2 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)
        ])

        # === Heatmap ===

        self.conv3 = BlazeBlock(block_num=3, channel=48)
        self.conv4 = BlazeBlock(block_num=4, channel=96)
        self.conv5 = BlazeBlock(block_num=5, channel=192)
        self.conv6 = BlazeBlock(block_num=6, channel=288)

        self.conv7a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv7b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu")
        ])

        self.conv8a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear")
        self.conv8b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu")
        ])

        self.conv9a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear")
        self.conv9b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu")
        ])

        self.conv10a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv10b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu")
        ])

        self.conv11 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=1, activation="relu"),
            tf.keras.layers.Conv2D(
                filters=self.num_keypoints, kernel_size=3, padding="same", activation=None) # -> Heatmap output
        ])

        # === Regression ===

        #  In: 1, 64, 64, 48)
        self.conv12a = BlazeBlock(block_num=4, channel=96, name_prefix="regression_conv12a_")    # input res: 64
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv12b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=96, kernel_size=1, activation="relu", name="regression_conv12b_conv1x1")
        ], name="regression_conv12b")

        self.conv13a = BlazeBlock(block_num=5, channel=192, name_prefix="regression_conv13a_")   # input res: 32
        self.conv13b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv13b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=192, kernel_size=1, activation="relu", name="regression_conv13b_conv1x1")
        ], name="regression_conv13b")

        self.conv14a = BlazeBlock(block_num=6, channel=288, name_prefix="regression_conv14a_")   # input res: 16
        self.conv14b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv14b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=288, kernel_size=1, activation="relu", name="regression_conv14b_conv1x1")
        ], name="regression_conv14b")

        self.conv15 = tf.keras.models.Sequential([
            BlazeBlock(block_num=7, channel=288, channel_padding=0, name_prefix="regression_conv15a_"),
            BlazeBlock(block_num=7, channel=288, channel_padding=0, name_prefix="regression_conv15b_")
        ], name="regression_conv15")

        self.conv16 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=3*self.num_keypoints, kernel_size=2, activation=None),
            tf.keras.layers.Reshape((3*self.num_keypoints, 1), name="regression_final_dense")
        ], name="joints")

        self.relu6 = layers.ReLU(max_value=6)

    def build_model(self, model_type, input_size):

        input_x = tf.keras.layers.Input(shape=(input_size, input_size, 3))

        # Block 1
        # In: 1x256x256x3
        x = self.conv1(input_x)

        # Block 2
        # In: 1x128x128x24
        x = x + self.conv2_1(x)
        x = tf.keras.activations.relu(x)

        # Block 3
        # In: 1x128x128x24
        x = x + self.conv2_2(x)
        y0 = tf.keras.activations.relu(x)

        # === Heatmap ===

        # In: 1, 128, 128, 24
        y1 = self.conv3(y0)
        y2 = self.conv4(y1)
        y3 = self.conv5(y2)
        y4 = self.conv6(y3)

        x = self.conv7a(y4) + self.conv7b(y3)
        x = self.conv8a(x) + self.conv8b(y2)
        # In: 1, 32, 32, 96
        x = self.conv9a(x) + self.conv9b(y1)
        # In: 1, 64, 64, 48
        y = self.conv10a(x) + self.conv10b(y0)
        y = self.conv11(y)
        y = self.relu6(y)

        # In: 1, 128, 128, 8
        #heatmap = tf.keras.layers.Activation("sigmoid", name="heatmap")(y)
        # === Regression ===
        return Model(inputs=input_x, outputs=y)
