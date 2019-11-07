from .aug_pipelines import medium_aug


dataloaders_conf = {
    'train': {
        'batch_size': 32,
        'num_workers': 8,
        'shuffle': True,
        'augmentation_pipeline': medium_aug(original_height=128, original_width=128),
    },
   'val': {
        'batch_size': 32,
        'num_workers': 8,
        'shuffle': False,
        'augmentation_pipeline': None,
    },
    'test': {
        'batch_size': 32,
        'num_workers': 8,
        'shuffle': True,
        'augmentation_pipeline': None,
    },
}

model_conf = {
    'device': 'cuda:0',
    'weight': [1, 10],
    'loss': [('BCE', 0.5), ('Dice_log', 0.5)],
#     'device': 'cpu',
    'opt_type': 'AdamW',
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'amsgrad': False,
    'nesterov': False,
    'momentum': 0.9,
    'centered': False,
    'min_lr': 1e-6,
    'factor': 0.5,
    'patience': 5,
}

train_conf = {
    'num_epochs': 200,
    'device': 'cuda:0',
#     'device': 'cpu',
}