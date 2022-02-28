import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    # Random generator seed
    config.seed = 41

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
    config.aug_configs.saturation = 0.2
    config.aug_configs.hue = 0.1
    config.aug_configs.gaussian_prob = [0.0, 0.0]
    config.aug_configs.solarization_prob = [0.0, 0.2]
    config.aug_configs.mean = [0.4914, 0.4822, 0.4465]
    config.aug_configs.std = [0.247, 0.243, 0.261]

    # Accelerator
    config.accelerator_configs = ml_collections.ConfigDict()
    config.accelerator_configs.use_gpu = False
    config.accelerator_configs.num_cores = 8
    config.accelerator_configs.device_name = 'local'
    
    # Model
    config.model = 'byol'
    config.model_configs = ml_collections.ConfigDict()
    config.model_configs.backbone = 'resnet18'
    config.model_configs.output_dim = 512
    config.model_configs.proj_hidden_dim = 2048
    config.model_configs.pred_hidden_dim = 2048
    config.model_configs.num_proj_layers = 1
    config.model_configs.num_pred_layers = 1

    # Training
    config.optimizer = 'adamw'
    config.ft_optimizer = 'adamw'
    config.optimizer_configs = ml_collections.ConfigDict()
    config.optimizer_configs.weight_decay = 1e-06
    config.ft_optimizer_configs = ml_collections.ConfigDict()
    config.ft_optimizer_configs.weight_decay = 1e-06
    config.batch_size = 256
    config.eval_batch_size = 256
    config.online_ft = True
    config.linear_eval = True
    config.learning_rate = 1e-03
    config.ft_learning_rate = 2e-04
    config.train_epochs = 1000
    config.warmup_epochs = 10

    # Loggings
    # config.history_dir = 'examples/training_history'
    # config.weights_dir = 'examples/weights'
    config.history_dir = None
    config.weights_dir = None
    config.weights_filename = 'byol_weights.hdf5'
    config.history_filename = 'byol_results.csv'
    config.callback_configs = ml_collections.ConfigDict()
    config.callback_configs.init_tau = 0.99
    config.callback_configs.final_tau = 1.0
    
    return config