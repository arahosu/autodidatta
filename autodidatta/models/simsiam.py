import tensorflow as tf
import tensorflow.keras.layers as tfkl

from autodidatta.models.base import BaseModel
from autodidatta.models.networks.mlp import projector_head, predictor_head


class SimSiam(BaseModel):

    def __init__(self,
                 backbone,
                 num_proj_layers=1,
                 num_pred_layers=1,
                 proj_hidden_dim=2048,
                 pred_hidden_dim=512,
                 output_dim=128,
                 custom_projector=None,
                 custom_predictor=None,
                 classifier=None,
                 train_projector=True):

        super(SimSiam, self).__init__(
            backbone=backbone,
            projector=custom_projector,
            predictor=custom_predictor,
            classifier=classifier
        )

        if self.distribute_strategy.num_replicas_in_sync > 1:
            global_bn = True
        else:
            global_bn = False

        if custom_projector is None:
            self.projector = projector_head(
                hidden_dim=proj_hidden_dim,
                output_dim=output_dim,
                num_layers=num_proj_layers,
                batch_norm_output=True,
                global_bn=global_bn
            )
        else:
            self.projector = custom_projector

        if custom_predictor is None:
            self.projector = predictor_head(
                hidden_dim=pred_hidden_dim,
                output_dim=output_dim,
                num_layers=num_pred_layers,
                global_bn=global_bn
            )
        else:
            self.projector = custom_projector 
        
        self.train_projector = train_projector
    
    def compile(self, loss_fn=None, ft_optimizer=None, **kwargs):
        super(SimSiam, self).compile(ft_optimizer=ft_optimizer, **kwargs)
        if loss_fn is None:
            self.loss_fn = tf.keras.losses.cosine_similarity
        else:
            self.loss_fn = loss_fn

    def shared_step(self, data, training):
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data
        num_channels = int(x.shape[-1] // 2)

        xi = x[..., :num_channels]
        xj = x[..., num_channels:]

        feat_i = self.backbone(xi, training=training)
        feat_j = self.backbone(xj, training=training)

        zi = self.projector(feat_i, training=training)
        zj = self.projector(feat_j, training=training)

        pi = self.predictor(zi, training=training)
        pj = self.predictor(zj, training=training)

        loss = self.loss_fn(pi, tf.stop_gradient(zj)) / 2
        loss += self.loss_fn(pj, tf.stop_gradient(zi)) / 2

        return loss

    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss = self.shared_step(data, training=True)
        if self.train_projector:
            trainable_variables = self.backbone.trainable_variables + \
                self.projector.trainable_variables + \
                self.predictor.trainable_variables
        else:
            trainable_variables = self.backbone.trainable_variables + \
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