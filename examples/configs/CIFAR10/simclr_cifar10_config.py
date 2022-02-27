import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    # Random generator seed
    config.seed = 42

    # Dataset
    config.dataset_name = 'cifar10'
    config.dataset_dir = None
    config.train_split = 'train'
    config.eval_split = 'test'
    config.dtype_str = 'mixed_bfloat16'
    
    # Augmentation
    config.aug_configs = ml_collections.ConfigDict()
    config.aug_configs.brightness = 0.4
    config.aug_configs.contrast = 0.4
    config.aug_configs.saturation = 0.4
    config.aug_configs.hue = 0.1
    config.aug_configs.gaussian_prob = [0.0, 0.0]
    config.aug_configs.solarization_prob = [0.0, 0.0]
    config.aug_configs.mean = [0., 0., 0.]
    config.aug_configs.std = [1., 1., 1.]

    # Accelerator
    config.accelerator_configs = ml_collections.ConfigDict()
    config.accelerator_configs.use_gpu = False
    config.accelerator_configs.num_cores = 8
    config.accelerator_configs.device_name = 'local'
    
    # Model
    config.model = 'simclr'
    config.model_configs = ml_collections.ConfigDict()
    config.model_configs.backbone = 'resnet18'
    config.model_configs.loss_temperature = 0.2
    config.model_configs.output_dim = 256
    config.model_configs.proj_hidden_dim = 512
    config.model_configs.num_proj_layers = 1

    # Training
    config.optimizer = 'adamw'
    config.ft_optimizer = 'adamw'
    config.optimizer_configs = ml_collections.ConfigDict()
    config.optimizer_configs.weight_decay = 1e-06
    # config.optimizer_configs.clipnorm = 1.0
    config.ft_optimizer_configs = ml_collections.ConfigDict()
    config.ft_optimizer_configs.weight_decay = 1e-06
    # config.ft_optimizer_configs.clipnorm = 1.0
    config.batch_size = 512
    config.eval_batch_size = 256
    config.online_ft = True
    config.linear_eval = True
    config.learning_rate = 1e-03
    config.ft_learning_rate = 2e-04
    config.train_epochs = 1000
    config.warmup_epochs = 10

    # Loggings
    config.history_dir = 'examples/training_history'
    config.weights_dir = 'examples/weights'
    config.weights_filename = 'simclr_weights.hdf5'
    config.history_filename = 'simclr_results.csv'
    config.callback_configs = ml_collections.ConfigDict()
    
    return config