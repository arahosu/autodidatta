import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Augmentation
    config.aug_configs = ml_collections.ConfigDict()
    config.aug_configs.brightness = 0.4
    config.aug_configs.contrast = 0.4
    config.aug_configs.saturation = 0.4
    config.aug_configs.hue = 0.1
    config.aug_configs.gaussian_prob = [0.0, 0.0]
    config.aug_configs.solarization_prob = [0.0, 0.0]
    config.aug_configs.mean = [0.4914, 0.4822, 0.4465]
    config.aug_configs.std = [0.247, 0.243, 0.261]
    
    # Model
    config.model = 'simclr'
    config.model_configs = ml_collections.ConfigDict()
    config.model_configs.backbone = 'resnet18'
    config.model_configs.output_dim = 512
    config.model_configs.proj_hidden_dim = 2048
    config.model_configs.num_proj_layers = 1
    config.model_configs.loss_temperature = 0.2

    # Training
    config.optimizer = 'adamw'
    config.base_learning_rate = 1e-03
    config.scheduler_configs = ml_collections.ConfigDict()
    config.scheduler_configs.warmup_epochs = 10
    config.scheduler_configs.learning_rate_scaling = None
    config.optimizer_configs = ml_collections.ConfigDict()
    config.optimizer_configs.weight_decay = 1e-06
    
    config.ft_optimizer = 'adamw'
    config.ft_learning_rate = 2e-04
    config.ft_optimizer_configs = ml_collections.ConfigDict()
    config.ft_optimizer_configs.weight_decay = 1e-06
    
    # Callback
    config.callback_configs = ml_collections.ConfigDict()
    
    return config