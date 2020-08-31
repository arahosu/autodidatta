import tensorflow as tf
import numpy as np

from self_supervised.TF2.model.basemodel import Resnet3DBuilder

SHAPE = (32,32,32,8)


def load_data():
    toy = np.random.rand(32,32,32,8)
    toy_samples = tf.data.Dataset.from_tensor_slices([tf.constant(toy) for i in range(5)])
    toy_labels = tf.data.Dataset.from_tensor_slices([tf.constant(toy) for i in range(5)])
    dataset = tf.data.Dataset.zip((toy_samples, toy_labels))
    return (toy_samples, toy_labels)


def make_backbone(input_spec: tf.TensorSpec, output_spec: tf.TensorSpec):
    # input_tensor = tf.keras.Input(shape=element_spec.shape[0:-1])
    # return tf.keras.applications.ResNet50(
    #     include_top=False, weights=None, input_tensor=input_tensor, input_shape=element_spec.shape[0:-1],
    #     pooling=None, classes=element_spec.shape[-1]
    # )
    return Resnet3DBuilder.build_resnet_50(
                input_shape=element_spec.shape,
                num_outputs=len(output_spec.shape),
                reg_factor=1e-4)


def create_model(input_spec: tf.TensorSpec, 
                 output_spec: tf.TensorSpec):
    backbone = make_backbone(input_spec, output_spec)
    # lr_rate = LearningRateSchedule()

    # model_fn, model_args = select_model()
    model = backbone

    # with strategy.scope():
    #     model = model_fn(*model_args)
    #     model.compile
    model.compile()

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


if __name__=="__main__":
    # specify parameters
    # strategy = setup_accelerator()

    # load data 
    data = load_data()  #aug_options)
    
    # specify model
    model = create_model(
                input_spec=data[0].element_spec, 
                output_spec=data[1].element_spec)  #model_opts)

    callbacks = tf.keras.callbacks.TensorBoard()

    # train
    is_train = True
    if is_train:  # FLAGS.is_train:
        history = model.fit(
            x=data[0], y=data[1], batch_size=1, epochs=5, callbacks=callbacks, steps_per_epoch=4)

    else:  # lif not FLAGS.is_train:
        model.load_weights()
        model.evaluate()
