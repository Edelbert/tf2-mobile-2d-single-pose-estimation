# Copyright 2020 Doyoung Gwak (tucan.dev@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import datetime



from data_loader.datasets import MHPDataset
from data_loader.data_loader import MHPLoader
from data_loader.dataset_augment import Augmentation

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.random.set_seed(3)
import numpy as np

from common import get_time_and_step_interval

print("tensorflow version   :", tf.__version__) # 2.1.0
print("keras version        :", tf.keras.__version__) # 2.2.4-tf

import sys
import getopt
from configparser import ConfigParser

"""
python train.py --dataset_config=config/dataset/coco2017-gpu.cfg --experiment_config=config/training/experiment01.cfg
python train.py --dataset_config=config/dataset/ai_challenger-gpu.cfg --experiment_config=config/training/experiment01.cfg
"""

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "d:e:", ["dataset_config=", "experiment_config="])
except getopt.GetoptError:
    print('train_hourglass.py --dataset_config <inputfile> --experiment_config <outputfile>')
    sys.exit(2)

dataset_config_file_path = "config/dataset/coco2017-gpu.cfg"
experiment_config_file_path = "config/training/experiment01.cfg"
for opt, arg in opts:
    if opt == '-h':
        print('train_middlelayer.py --dataset_config <inputfile> --experiment_config <outputfile>')
        sys.exit()
    elif opt in ("-d", "--dataset_config"):
        dataset_config_file_path = arg
    elif opt in ("-e", "--experiment_config"):
        experiment_config_file_path = arg

parser = ConfigParser()

# get dataset config
print(dataset_config_file_path)
parser.read(dataset_config_file_path)
config_dataset = {}
for key in parser["dataset"]:
    config_dataset[key] = eval(parser["dataset"][key])

# get training config
print(experiment_config_file_path)
parser.read(experiment_config_file_path)

config_preproc = {}
if "preprocessing" in parser:
    for key in parser["preprocessing"]:
        config_preproc[key] = eval(parser["preprocessing"][key])
config_model = {}
for key in parser["model"]:
    config_model[key] = eval(parser["model"][key])
config_extra = {}
if "extra" in parser:
    for key in parser["extra"]:
        config_extra[key] = eval(parser["extra"][key])
config_training = {}
if "training" in parser:
    for key in parser["training"]:
        config_training[key] = eval(parser["training"][key])
config_output = {}
if "output" in parser:
    for key in parser["output"]:
        config_output[key] = eval(parser["output"][key])

dataset_root_path = config_dataset["dataset_root_path"]  # "/Volumes/tucan-SSD/datasets"
dataset_directory_name = config_dataset["dataset_directory_name"]  # "coco_dataset"
dataset_path = os.path.join(dataset_root_path, dataset_directory_name)

output_root_path = config_output["output_root_path"]  # "/home/outputs"  # "/Volumes/tucan-SSD/ml-project/outputs"
output_experiment_name = config_output["experiment_name"]  # "experiment01"
sub_experiment_name = config_output["sub_experiment_name"]  # "basic"
current_time = datetime.datetime.now().strftime("%m%d%H%M")
model_name = config_model["model_name"]  # "simplepose"
model_subname = config_model["model_subname"]
model_backbone_name = config_model.get('backbone_name')
output_name = f"{current_time}_{model_name}_{sub_experiment_name}"
output_path = os.path.join(output_root_path, output_experiment_name, dataset_directory_name)
output_log_path = os.path.join(output_path, "logs", output_name)

# =================================================
# ============== prepare training =================
# =================================================

train_summary_writer = tf.summary.create_file_writer(output_log_path)

@tf.function
def train_step(model, images, labels):
    #print(images, labels)
    with tf.GradientTape() as tape:
        model_output = model(images)
        predictions_layers = model_output
        if isinstance(predictions_layers, list):
            losses = [loss_object(labels, predictions) for predictions in predictions_layers]
            total_loss = tf.math.add_n(losses) / images.shape[0]
            loss_val = losses[-1]
        else:
            #print(labels.shape, predictions_layers.shape)
            tf.print(tf.math.reduce_mean(images), tf.math.reduce_mean(predictions_layers))
            #tf.print(predictions_layers)
            total_loss = loss_object(labels, predictions_layers) 
            loss_val =total_loss
        

    max_val = tf.math.reduce_max(predictions_layers[-1])

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(total_loss)
    return total_loss, loss_val, max_val

from save_result_as_image import save_result_image

def val_step(step, images, heamaps):
    #images, labels = images
    predictions = model(images, training=False)
    predictions = np.array(predictions)
    save_image_results(step, images, heamaps, predictions)

from evaluate import calculate_total_pckh

@tf.function
def valid_step(model, images, labels):
    predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)
    valid_loss(v_loss)
    # valid_accuracy(labels, predictions)
    return v_loss

def save_image_results(step, images, true_heatmaps, predicted_heatmaps):
    val_image_results_directory = "val_image_results"

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, output_name)):
        os.mkdir(os.path.join(output_path, output_name))
    if not os.path.exists(os.path.join(output_path, output_name, val_image_results_directory)):
        os.mkdir(os.path.join(output_path, output_name, val_image_results_directory))
    for i in range(images.shape[0]):
        image = images[i, :, :, :]
        heamap = true_heatmaps[i, :, :, :]
        if isinstance(predicted_heatmaps, list):
            prediction = predicted_heatmaps[-1][i, :, :, :]
        else:
            prediction = predicted_heatmaps[i, :, :, :]

        # result_image = display(i, image, heamap, prediction)
        result_image_path = os.path.join(output_path, output_name, val_image_results_directory, f"result{i}-{step:0>6d}.jpg")
        save_result_image(result_image_path, image, heamap, prediction, title=f"step:{int(step/1000)}k")
        print("val_step: save result image on \"" + result_image_path + "\"")

def save_model(model, step=None, label=None, post_label=None):
    saved_model_directory = "saved_model"
    if step is not None:
        saved_model_directory = saved_model_directory + f"-{step:0>6d}"
    if label is not None:
        saved_model_directory = saved_model_directory + "-" + label
    if post_label is not None:
        saved_model_directory = saved_model_directory + "-" + post_label

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, output_name)):
        os.mkdir(os.path.join(output_path, output_name))
    if not os.path.exists(os.path.join(output_path, output_name, saved_model_directory)):
        os.mkdir(os.path.join(output_path, output_name, saved_model_directory))

    saved_model_path = os.path.join(output_path, output_name, saved_model_directory)

    print("-"*20 + " MODEL SAVE!! " + "-"*20)
    print("saved model path: " + saved_model_path)
    model.save(saved_model_path)
    print("-"*18 + " MODEL SAVE DONE!! " + "-"*18)

    return saved_model_path

if __name__ == '__main__':
    # ================================================
    # ============= load hyperparams =================
    # ================================================
    # config_dataset = ...
    # config_model = ...
    # config_output = ...
    
    #     
    # strategy = tf.distribute.MirroredStrategy()
    strategy = None

    # ================================================
    # =============== load dataset ===================
    # ================================================
    from data_loader.data_loader import DataLoader

    config_dataset['batch_size'] = config_training['batch_size']
    config_dataset['num_keypoints'] = config_preproc['num_keypoints']
    config_dataset['in_height'] = config_model['input_height']
    config_dataset['in_width'] = config_model['input_width']
    config_dataset['out_height'] = config_model['output_height']
    config_dataset['out_width'] = config_model['output_width']
    config_dataset['heatmap_std'] = config_preproc['heatmap_std']

    # dataloader instance gen
    dataset_name = config_dataset['dataset_name']
    train_images = config_dataset["train_images"]
    train_annotation = config_dataset["train_annotation"]
    train_images_dir_path = os.path.join(dataset_path, train_images)
    train_annotation_json_filepath = os.path.join(dataset_path, train_annotation)
    print(">> LOAD TRAIN DATASET FORM:", train_annotation_json_filepath)

    train_dataset = MHPDataset(train_annotation_json_filepath)

    steps_per_epoch = int(len(train_dataset) / config_dataset['batch_size'])

    augmentor = Augmentation()
    train_dataset = MHPLoader(train_dataset, augmentor, config = config_dataset, train = True)

    valid_images = config_dataset["valid_images"] if "valid_images" in config_dataset else None
    valid_annotation = config_dataset["valid_annotation"] if "valid_annotation" in config_dataset else None
    dataloader_valid = None
    if valid_images is not None:
        valid_images_dir_path = os.path.join(dataset_path, valid_images)
        valid_annotation_json_filepath = os.path.join(dataset_path, valid_annotation)
        print(">> LOAD VALID DATASET FORM:", valid_annotation_json_filepath)
        val_dataset =  MHPDataset(valid_annotation_json_filepath, shuffle=False)
        val_dataset = MHPLoader(val_dataset, augmentor = None, config = config_dataset, train = False)

    number_of_keypoints = config_preproc['num_keypoints']  # 17

    # train dataset

    #dataset_train = strategy.experimental_distribute_dataset(dataset_train)
    #dataset_valid = strategy.experimental_distribute_dataset(dataset_valid)

    # validation images
    #get val pics

    data = next(iter(val_dataset))
    val_images, val_heatmaps = data['image'], data['heatmap']

   # print(val_heatmaps)
    #print('images', val_images)
    #print('heatmaps', val_heatmaps)
    # ================================================
    # =============== build model ====================
    # ================================================
    from model_provider import get_model
    if strategy is not None:
        with strategy.scope():
            model = get_model(model_name=model_name,
                        model_subname=model_subname,
                        number_of_keypoints=number_of_keypoints,
                        config_extra=config_extra, 
                        backbone_name=model_backbone_name)
    else:
        model = get_model(model_name=model_name,
                        model_subname=model_subname,
                        number_of_keypoints=number_of_keypoints,
                        config_extra=config_extra, 
                        backbone_name=model_backbone_name,
                        input_size=config_dataset['in_height'],
                        weights=None)

    loss_object = tf.keras.losses.MeanSquaredError()
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = config_training["learning_rate"], decay_steps = 5000, decay_rate = config_training['decay_rate'])
    optimizer = tf.keras.optimizers.Adam(scheduler, epsilon=config_training["epsilon"])
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    valid_loss = tf.keras.metrics.Mean(name="valid_loss")
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")

    # ================================================
    # ============== train the model =================
    # ================================================

    num_epochs = config_training["number_of_epoch"]  # 550
    number_of_echo_period = config_training["period_echo"]  # 100
    number_of_validimage_period = 5000  # 1000
    number_of_modelsave_period = config_training["period_save_model"]  # 5000
    tensorbaord_period = config_training["period_tensorboard"]  # 100
    validation_period = 2  # 1000
    valid_check = False
    valid_pckh = config_training["valid_pckh"]  # True
    pckh_distance_ratio = config_training["pckh_distance_ratio"]  # 0.5

    step = 1
    # TRAIN!!
    get_time_and_step_interval(step, is_init=True)
    print('steps per epoch', steps_per_epoch)
    if steps_per_epoch < 300:
        steps_per_epoch *= 20

    for epoch in range(num_epochs):
        print("-" * 10 + " " + str(epoch + 1) + " EPOCH " + "-" * 10)
        for data in train_dataset:
            #print(len(data))
            images, heatmaps = data['image'], data['heatmap']
            #print(np.mean(images))
            #print(images.shape, heatmaps.shape)
            # print(images.shape)  # (32, 128, 128, 3)
            # print(heatmaps.shape)  # (32, 32, 32, 17)
            total_loss, last_layer_loss, max_val = train_step(model, images, heatmaps)
            #return 0
            step += 1

            if number_of_echo_period is not None and step % number_of_echo_period == 0:
                total_interval, per_step_interval = get_time_and_step_interval(step)
                echo_textes = []
                if step is not None:
                    echo_textes.append(f"step: {step}")
                if total_interval is not None:
                    echo_textes.append(f"total: {total_interval}")
                if per_step_interval is not None:
                    echo_textes.append(f"per_step: {per_step_interval}")
                if total_loss is not None:
                    echo_textes.append(f"total loss: {total_loss:.6f}")
                if last_layer_loss is not None:
                    echo_textes.append(f"last loss: {last_layer_loss:.6f}")
                if train_loss:
                    echo_textes.append('train loss for epoch: {:.6f}'.format(train_loss.result().numpy()))
                print(">> " + ", ".join(echo_textes))

            # validation phase
            if number_of_validimage_period is not None and step % number_of_validimage_period == 0:
                val_step(step, val_images, val_heatmaps)

            if number_of_modelsave_period is not None and step % number_of_modelsave_period == 0:
                saved_model_path = save_model(model, step=step)

                if valid_pckh:
                    # print("calcuate pckh")
                    pckh_score = calculate_total_pckh(saved_model_path=saved_model_path,
                                                      annotation_path=valid_annotation_json_filepath,
                                                      images_path=valid_images_dir_path,
                                                      distance_ratio=pckh_distance_ratio)
                    with train_summary_writer.as_default():
                        tf.summary.scalar(f'tflite-pckh@{pckh_distance_ratio:.1f}', pckh_score * 100, step=step)

            if tensorbaord_period is not None and step % tensorbaord_period == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("total_loss", total_loss.numpy(), step=step)
                    tf.summary.scalar("max_value - last_layer_loss", max_val.numpy(), step=step)
                    if last_layer_loss is not None:
                        tf.summary.scalar("last_layer_loss", last_layer_loss.numpy(), step=step)
            
            if  step % steps_per_epoch == 0:
                train_loss.reset_states()
                break

        # if not valid_check:
        #     continue

        # for v_images, v_heatmaps in dataloader_valid:
        #     v_loss = valid_step(model, sv_images, v_heatmaps)



    # last model save
    saved_model_path = save_model(model, step=step, label="final", post_label=f"pckh{pckh_score:.3f}")

    # last pckh
    pckh_score = calculate_total_pckh(saved_model_path=saved_model_path,
                                      annotation_path=valid_annotation_json_filepath,
                                      images_path=valid_images_dir_path,
                                      distance_ratio=pckh_distance_ratio)
    with train_summary_writer.as_default():
        tf.summary.scalar(f'pckh@{pckh_distance_ratio:.1f}_score', pckh_score * 100, step=step)
