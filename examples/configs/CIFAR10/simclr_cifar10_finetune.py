import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Augmentation
    config.mean = [0.4914, 0.4822, 0.4465]
    config.std = [0.247, 0.243, 0.261]
    
    # Model
    config.model = 'simclr'
    config.backbone = 'resnet18'

    # Training
    config.optimizer = 'adamw'
    config.learning_rate = 2e-04
    config.optimizer_configs = ml_collections.ConfigDict()
    config.optimizer_configs.weight_decay = 1e-06
    
    # Callback
    config.callback_configs = ml_collections.ConfigDict()
    
    return config