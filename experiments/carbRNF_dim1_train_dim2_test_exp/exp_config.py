data_conf = {
    'conf_name': 'carbRNF_dim1_train_dim2_test_exp',
    'stacks': [
        {
            'path': '../../data/carbRNF',
            'slice_train': (slice(None), slice(None), slice(230)),
            'slice_val': (slice(None), slice(None), slice(250, 470)),
        },
        {
            'path': '../../data/carb96558',
            'slice_test': (slice(None), slice(None), slice(490, None)),
        },
        {
            'path': '../../data/carb71',
            'slice_test': (slice(None), slice(None), slice(490, None)),
        },
        {
            'path': '../../data/carbRNF',
            'slice_test': (slice(None), slice(None), slice(470, None)),
        },
        {
            'path': '../../data/SPE_carb10_58_box3',
            'slice_test': (slice(None), slice(None), slice(280, None)),
        },
        {
            'path': '../../data/SoilAh-1',
            'slice_test': (slice(None), slice(None), slice(470, None)),
        },
        {
            'path': '../../data/SoilB-2',
            'slice_test': (slice(None), slice(None), slice(470, None)),
        },
        {
            'path': '../../data/TeTree_subset1',
            'slice_test': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../data/TiTree_subset2',
            'slice_test': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../data/Urna_22',
            'slice_test': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../data/Urna_30',
            'slice_test': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../data/Urna_34',
            'slice_test': (slice(None), slice(None), slice(470, None)),
        },
    ],
    'patches': {
        'train': (128, 1, 128),
        'val': (128, 1, 128),
        'test': (128, 128, 1),
    },
}