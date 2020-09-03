import tensorflow as tf
import numpy as np

import sys
# Yes this is awful but it lets modules from sibling directories be imported  https://docs.python.org/3/tutorial/modules.html#the-module-search-path
sys.path.insert(0,'./models') # 
sys.path.insert(0,'./') # 
print('sys.path', sys.path)

from models.basemodel import ResnetBuilder
# import models.basemodel as basemodel

SHAPE = (32,32,32,8)


def load_data():
    toy = np.random.randint(0, 1, size=SHAPE)
    toy_samples = tf.data.Dataset.from_tensors([tf.constant(toy) for i in range(5)])
    toy_labels = tf.data.Dataset.from_tensors([tf.constant(toy) for i in range(5)])
    dataset = tf.data.Dataset.zip((toy_samples, toy_labels))
    return (toy_samples, toy_labels)


def make_backbone(input_shape: tf.TensorShape, 
                  output_shape: tf.TensorShape,
                  batch_size=None):
    # input_tensor = tf.keras.Input(shape=element_spec.shape[0:-1])
    # return tf.keras.applications.ResNet50(
    #     include_top=False, weights=None, input_tensor=input_tensor, input_shape=element_spec.shape[0:-1],
    #     pooling=None, classes=element_spec.shape[-1]
    # )
    return ResnetBuilder.build_resnet_50(
                input_shape=input_shape,
                num_outputs=output_shape,  #len(output_shape),
                batch_size=batch_size,
                reg_factor=1e-4)


def create_model(input_shape: tf.TensorShape,
                 output_shape: tf.TensorShape,
                 batch_size=None):
    backbone = make_backbone(input_shape, output_shape, batch_size)
    # lr_rate = LearningRateSchedule()

    # model_fn, model_args = select_model()
    model = backbone

    # with strategy.scope():
    #     model = model_fn(*model_args)
    #     model.compile
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # return model
    return model 


def select_model():
    if model_choice == 'SimCLR':
        return simclr
    else:
        return 0


def simclr():
    def train_step():
        # implement contrastive learning
        loss = 0
    # self.output_layer = define necessary steps, objective fxn 


if __name__ == "__main__":
    # specify parameters
    # strategy = setup_accelerator()

    # load data 
    data = load_data()  #aug_options)
    
    # specify model
    batch_size = data[0].element_spec.shape[0]
    model = create_model(
                input_shape=data[0].element_spec.shape[1:], 
                output_shape=data[1].element_spec.shape[1:],
                batch_size=batch_size)  #model_opts)

    callbacks = tf.keras.callbacks.TensorBoard()

    # train
    is_train = True
    if is_train:  # FLAGS.is_train:
        history = model.fit(
            x=tf.data.Dataset.zip((data[0], data[1])), batch_size=batch_size, epochs=5, callbacks=callbacks, steps_per_epoch=4)

    else:  # lif not FLAGS.is_train:
        model.load_weights()
        model.evaluate()
