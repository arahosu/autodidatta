import tensorflow as tf


class BaseModel(tf.keras.Model):

    def __init__(self,
                 backbone,
                 projector,
                 predictor=None,
                 classifier=None):
        
        super(BaseModel, self).__init__()

        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor
        self.classifier = classifier

    def build(self, input_shape):

        self.backbone.build(input_shape)
        self.projector.build(self.backbone.compute_output_shape(input_shape))

        if self.predictor is not None:
            self.predictor.build(
                self.projector.compute_output_shape(
                    self.backbone.compute_output_shape(input_shape)))

        if self.classifier is not None:
            self.classifier.build(
                self.backbone.compute_output_shape(input_shape))

        self.built = True
    
    def call(self, x, training=False):

        return self.backbone(x, training=training)
    
    def compile(self, loss_fn, ft_optimizer=None, **kwargs):
        super(BaseModel, self).compile(**kwargs)
        self.loss_fn = loss_fn
        if self.classifier is not None:
            assert ft_optimizer is not None, \
                'ft_optimizer should not be None if self.classifier is not \
                    None'
            self.ft_optimizer = ft_optimizer
    
    def compute_output_shape(self, input_shape):
        return self.backbone.compute_output_shape(input_shape)
    
    def shared_step(self, data, training=True):
        raise NotImplementedError("shared_step is not implemented in BaseModel")

    def finetune_step(self, data):
        x, y = data
        num_channels = int(x.shape[-1] // 2)
        view = x[..., :num_channels]

        with tf.GradientTape() as tape:
            features = self(view, training=True)
            y_pred = self.classifier(features, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        trainable_variables = self.classifier.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.ft_optimizer.apply_gradients(zip(grads, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
    
    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss = self.shared_step(data, training=True)
        
        if self.predictor is not None:
            trainable_variables = self.backbone.trainable_variables + \
                self.projector.trainable_variables + \
                self.predictor.trainable_variables
        else:
            trainable_variables = self.backbone.trainable_variables + \
                self.projector.trainable_variables

        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        if self.classifier is not None:
            self.finetune_step(data)
            metrics_results = {m.name: m.result() for m in self.metrics}
            results = {'similarity_loss': loss, **metrics_results}
        else:
            results = {'similarity_loss': loss}

        return results
    
    def test_step(self, data):

        if isinstance(data, tuple):
            x, y = data
        else:
            x = data

        num_channels = int(x.shape[-1] // 2)
        view = x[..., :num_channels]

        loss = self.shared_step(data, training=False)

        if self.classifier is not None:
            features = self.backbone(view, training=False)
            y_pred = self.classifier(features, training=False)
            _ = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred)
            metric_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': loss, **metric_results}
        else:
            return {'similarity_loss': loss}

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     options=None,
                     save_backbone_only=True):
        if save_backbone_only:
            weights = self.backbone.save_weights(
                filepath, overwrite, save_format, options)
        else:
            assert self.built, 'The model must first be built \
                                before its weights can be saved'
            weights = super(BaseModel, self).save_weights(
                filepath, overwrite, save_format, options)
        return weights