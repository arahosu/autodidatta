import tensorflow as tf
import tensorflow.keras.layers as tfkl

from autodidatta.models.base import BaseModel
from autodidatta.models.networks.mlp import projector_head
from autodidatta.utils.loss import nt_xent_loss_v2


class SimCLR(BaseModel):

    def __init__(self,
                 backbone,
                 num_proj_layers=1,
                 proj_hidden_dim=512,
                 output_dim=128,
                 custom_projector=None,
                 classifier=None,
                 loss_temperature=0.5):

        super(SimCLR, self).__init__(
            backbone=backbone,
            projector=custom_projector,
            predictor=None,
            classifier=classifier
        )

        if custom_projector is None:
            if self.distribute_strategy.num_replicas_in_sync > 1:
                global_bn = True
            else:
                global_bn = False

            self.projector = projector_head(
                hidden_dim=proj_hidden_dim,
                output_dim=output_dim,
                num_layers=num_proj_layers,
                batch_norm_output=True,
                global_bn=global_bn
            )
        else:
            self.projector = custom_projector
        
        self.loss_temperature = loss_temperature
    
    def compile(self, loss_fn=None, ft_optimizer=None,**kwargs):
        super(SimCLR, self).compile(ft_optimizer=ft_optimizer, **kwargs)
        if loss_fn is None:
            self.loss_fn = nt_xent_loss_v2
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

        zi = self.backbone(xi, training=training)
        zj = self.backbone(xj, training=training)

        if self.projector is not None:
            zi = self.projector(zi, training=training)
            zj = self.projector(zj, training=training)

        zi = tf.math.l2_normalize(zi, axis=-1)
        zj = tf.math.l2_normalize(zj, axis=-1)

        loss = self.loss_fn(
            zi, zj,
            temperature=self.loss_temperature,
            strategy=self.distribute_strategy)

        return loss
