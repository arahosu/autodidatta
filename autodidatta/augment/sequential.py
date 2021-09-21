from tensorflow.keras import Sequential


class Augment(Sequential):

    def __init__(self,
                 layers=None,
                 name=None,
                 **kwargs):
        super(Augment, self).__init__(
            layers=layers,
            name=name,
            **kwargs
            )

    def call(self, image, seg=None, training=False):

        # Handle corner cases where self.layers is empty
        aug_image, aug_seg = image, seg

        for layer in self.layers:
            aug_image, aug_seg = layer.apply(
                image=image, seg=seg, training=training)
            image, seg = aug_image, aug_seg
        if seg is not None:
            return (aug_image, aug_seg)
        else:
            return aug_image
