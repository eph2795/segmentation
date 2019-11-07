data_conf = {
    'conf_name': 'carb96558',
    'stacks': [
        {
            'path': '../../data/carb96558',
            'slice_train': (slice(None), slice(None), slice(230)),
            'slice_val': (slice(None), slice(None), slice(250, 470)),
        },
        {
            'path': '../../data/SoilB-2',
            'slice_train': (slice(None), slice(None), slice(230)),
            'slice_val': (slice(None), slice(None), slice(240, 460)),
        },
        {
            'path': '../../data/Urna_22',
            'slice_train': (slice(None), slice(None), slice(220)),
            'slice_val': (slice(None), slice(None), slice(245, 455)),
        },
        {
            'path': '../../data/carb96558',
            'slice_test': (slice(None), slice(None), slice(490, None)),
        },
        {
            'path': '../../data/carb71',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
        {
            'path': '../../data/carbRNF',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
        {
            'path': '../../data/SPE_carb10_58_box3',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
        {
            'path': '../../data/SoilAh-1',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
        {
            'path': '../../data/SoilB-2',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
        {
            'path': '../../data/TeTree_subset1',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
        {
            'path': '../../data/TiTree_subset2',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
        {
            'path': '../../data/Urna_22',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
        {
            'path': '../../data/Urna_30',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
        {
            'path': '../../data/Urna_34',
            'slice_test': (slice(None), slice(None), slice(None)),
        },
    ],
    'patches': {
        'train': (128, 128, 1),
        'val': (128, 128, 1),
        'test': (128, 128, 1),
    },
}