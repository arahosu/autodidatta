import math

import tensorflow as tf
import tensorflow.keras.layers as tfkl

from autodidatta.models.base import BaseModel
from autodidatta.models.networks.mlp import projector_head, predictor_head
from autodidatta.utils.loss import byol_loss


class BYOLMAWeightUpdate(tf.keras.callbacks.Callback):

    def __init__(self,
                 max_steps,
                 init_tau=0.99,
                 final_tau=1.0,
                 train_projector=True):
        super(BYOLMAWeightUpdate, self).__init__()

        assert abs(init_tau) <= 1.
        assert abs(final_tau) <= 1. and init_tau <= final_tau

        self.max_steps = max_steps
        self.init_tau = init_tau
        self.current_tau = init_tau
        self.final_tau = final_tau
        self.global_step = 0
        self.train_projector = train_projector

    def on_train_batch_end(self, batch, logs=None):
        self.update_weights()
        self.current_tau = self.update_tau()
        self.global_step += 1

    def update_tau(self):
        return self.final_tau - (self.final_tau - self.init_tau) * \
            (math.cos(math.pi * self.global_step / self.max_steps) + 1) / 2

    @tf.function
    def update_weights(self):
        for online_layer, target_layer in zip(
            self.model.backbone.layers,
            self.model.target_backbone.layers):
            if hasattr(target_layer, 'kernel'):
                target_layer.kernel.assign(self.current_tau * target_layer.kernel 
                                           + (1 - self.current_tau) * online_layer.kernel)
            if hasattr(target_layer, 'bias'):
                target_layer.bias.assign(self.current_tau * target_layer.bias 
                                         + (1 - self.current_tau) * online_layer.bias)
            if hasattr(target_layer, 'gamma'):
                target_layer.gamma.assign(self.current_tau * target_layer.gamma 
                                         + (1 - self.current_tau) * online_layer.gamma)
            if hasattr(target_layer, 'beta'):
                target_layer.beta.assign(self.current_tau * target_layer.beta 
                                         + (1 - self.current_tau) * online_layer.beta)
                

        if self.train_projector:
            for online_layer, target_layer in zip(
                self.model.projector.layers,
                self.model.target_projection.layers):
                if hasattr(target_layer, 'kernel'):
                    target_layer.kernel.assign(self.current_tau * 
                    target_layer.kernel + (1 - self.current_tau) * 
                    online_layer.kernel)
                if hasattr(target_layer, 'bias'):
                    if target_layer.bias is not None:
                        target_layer.bias.assign(self.current_tau * 
                        target_layer.bias + (1 - self.current_tau) * 
                        online_layer.bias)
                if hasattr(target_layer, 'gamma'):
                    target_layer.gamma.assign(self.current_tau * target_layer.gamma 
                    + (1 - self.current_tau) * online_layer.gamma)
                if hasattr(target_layer, 'beta'):
                    target_layer.beta.assign(self.current_tau * target_layer.beta 
                    + (1 - self.current_tau) * online_layer.beta)


class BYOL(BaseModel):

    def __init__(self,
                 backbone,
                 num_proj_layers=1,
                 num_pred_layers=1,
                 proj_hidden_dim=2048,
                 pred_hidden_dim=512,
                 output_dim=128,
                 custom_projector=None,
                 custom_predictor=None,
                 classifier=None):

        super(BYOL, self).__init__(
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

        self.target_backbone = tf.keras.models.clone_model(self.backbone)
        self.target_projection = tf.keras.models.clone_model(self.projector)

    def build(self, input_shape):

        self.backbone.build(input_shape)
        self.projector.build(
            self.backbone.compute_output_shape(input_shape))
        self.predictor.build(
            self.projector.compute_output_shape(
                self.backbone.compute_output_shape(input_shape)))
        
        self.target_backbone.build(input_shape)
        self.target_projection.build(
            self.target_backbone.compute_output_shape(input_shape))

        if self.classifier is not None:
            self.classifier.build(
                self.backbone.compute_output_shape(input_shape))

        self.built = True
    
    def compile(self, loss_fn=None, ft_optimizer=None, **kwargs):
        super(BYOL, self).compile(ft_optimizer=ft_optimizer, **kwargs)
        if loss_fn is None:
            self.loss_fn = byol_lossss
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

        zi = self.predictor(
            self.projector(
            self.backbone(xi, training),
            training),
            training)

        zj = self.predictor(
            self.projector(
                self.backbone(xj, training),
                training),
                training)

        pi = self.target_projection(
            self.target_backbone(xi, training),
            training)
        
        pj = self.target_projection(
            self.target_backbone(xj, training),
            training)

        loss = self.loss_fn(pi, zj, self.distribute_strategy)
        loss += self.loss_fn(pj, zi, self.distribute_strategy)

        return loss