import tensorflow as tf
from sss.vgg import VGG_UNet_Encoder, VGG_UNet_Decoder


class SimSiam(tf.keras.Model):

    def __init__(self,
                 backbone,
                 projection,
                 predictor,
                 classifier=None):

        super(SimSiam, self).__init__()

        self.backbone = backbone
        self.projection = projection
        self.predictor = predictor
        self.classifier = classifier

    def build(self, input_shape):

        self.backbone.build(input_shape)
        self.projection.build(self.backbone.compute_output_shape(input_shape))
        self.predictor.build(
            self.projection.compute_output_shape(
                self.backbone.compute_output_shape(input_shape)))

        if self.classifier is not None:
            self.classifier.build(
                self.backbone.compute_output_shape(input_shape))

        self.built = True

    def call(self, x, training=False):

        result = self.backbone(x, training=training)

        return result

    def compile(self, loss_fn, ft_optimizer=None, **kwargs):
        super(SimSiam, self).compile(**kwargs)
        self.loss_fn = loss_fn
        if self.classifier is not None:
            assert ft_optimizer is not None, \
                'ft_optimizer should not be None if self.classifier is not \
                    None'
            self.ft_optimizer = ft_optimizer

    def compute_output_shape(self, input_shape):

        current_shape = self.backbone.compute_output_shape(input_shape)

        return current_shape

    def shared_step(self, data, training):

        x, _ = data
        num_channels = int(x.shape[-1] // 2)

        xi = x[..., :num_channels]
        xj = x[..., num_channels:]

        zi = self.backbone(xi, training=training)
        zj = self.backbone(xj, training=training)

        zi = self.projection(zi, training=training)
        zj = self.projection(zj, training=training)

        pi = self.predictor(zi, training=training)
        pj = self.predictor(zj, training=training)

        loss = self.loss_fn(pi, tf.stop_gradient(zj)) / 2
        loss += self.loss_fn(pj, tf.stop_gradient(zi)) / 2

        return loss

    def finetune_step(self, data):
        x, y = data
        num_channels = int(x.shape[-1] // 2)
        view = x[..., :num_channels]

        if len(y.shape) > 2:
            num_classes = int(y.shape[-1] // 2)
            y = y[..., :num_classes]

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
        trainable_variables = self.backbone.trainable_variables + \
            self.projection.trainable_variables + \
            self.predictor.trainable_variables
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
        x, y = data
        num_channels = int(x.shape[-1] // 2)
        view = x[..., :num_channels]

        if len(y.shape) > 2:
            num_classes = int(y.shape[-1] // 2)
            y = y[..., :num_classes]
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


class SimSiam_UNet(tf.keras.Model):

    def __init__(self,
                 input_shape,
                 projection,
                 predictor,
                 classifier=None,
                 finetune_decoder_only=False):

        super(SimSiam_UNet, self).__init__()

        # Model specification
        self.encoder = VGG_UNet_Encoder(input_shape)
        output_shape = self.encoder.output_shape
        output_shape = [tuple(ele for ele in sub if ele is not None) for sub in output_shape]
        self.decoder = VGG_UNet_Decoder(output_shape)
        self.projection = projection
        self.predictor = predictor
        self.classifier = classifier

        # Training option specification
        self.finetune_decoder_only = finetune_decoder_only

    def call(self, x, training=False):

        x1, x2, x3, x4, x5 = self.encoder(x, training=training)
        return self.decoder([x1, x2, x3, x4, x5], training=training)

    def compile(self, loss_fn, ft_optimizer=None, **kwargs):
        super(SimSiam_UNet, self).compile(**kwargs)
        self.loss_fn = loss_fn
        assert ft_optimizer is not None, \
            'ft_optimizer should not be None if self.classifier is not \
                None'
        self.ft_optimizer = ft_optimizer

    def shared_step(self, data, training):

        x, _ = data
        num_channels = int(x.shape[-1] // 2)

        xi = x[..., :num_channels]
        xj = x[..., num_channels:]

        _, _, _, _, zi = self.encoder(xi, training=training)
        _, _, _, _, zj = self.encoder(xj, training=training)

        zi = self.projection(zi, training=training)
        zj = self.projection(zj, training=training)

        pi = self.predictor(zi, training=training)
        pj = self.predictor(zj, training=training)

        loss = self.loss_fn(pi, tf.stop_gradient(zj)) / 2
        loss += self.loss_fn(pj, tf.stop_gradient(zi)) / 2

        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.shared_step(data, training=True)
        trainable_variables = self.encoder.trainable_variables + \
            self.projection.trainable_variables + \
            self.predictor.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        if self.classifier is not None:
            self.finetune_step(data)
            metrics_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': loss, **metrics_results}
        else:
            return {'similarity_loss': loss}

    def finetune_step(self, data):

        x, y = data
        num_channels = int(x.shape[-1] // 2)
        view = x[..., :num_channels]
        if len(y.shape) > 2:
            num_classes = int(y.shape[-1] // 2)
            y = y[..., :num_classes]

        with tf.GradientTape() as tape:
            features = self.encoder(view, training=True)
            features = self.decoder(features, training=True)
            y_pred = self.classifier(features, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        if self.finetune_decoder_only:
            trainable_variables = self.classifier.trainable_variables + \
                self.decoder.trainable_variables
        else:
            trainable_variables = self.classifier.trainable_variables + \
                self.decoder.trainable_variables + \
                self.encoder.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.ft_optimizer.apply_gradients(zip(grads, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

    def test_step(self, data):

        sim_loss = self.shared_step(data, training=False)
        if self.classifier is not None:
            x, y = data
            num_channels = int(x.shape[-1] // 2)
            view = x[..., :num_channels]
            if len(y.shape) > 2:
                num_classes = int(y.shape[-1] // 2)
                y = y[..., :num_classes]
            features = self.encoder(view, training=False)
            features = self.decoder(features, training=False)
            y_pred = self.classifier(features, training=False)
            _ = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred)
            metric_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': sim_loss, **metric_results}
        else:
            return {'loss': sim_loss}
