# called by tools/make_sidebar.py

# This dict defines the structure:

sidebar_d = {
    'Getting started': {
        'Installation': 'https://github.com/fastai/fastai/blob/master/README.md#installation',
        'Troubleshooting': 'https://docs-dev.fast.ai/troubleshoot',
    },
    'Training': {
        'Overview': 'training',
        'basic_train': 'basic_train',
        'train': 'train',
        'metrics': 'metrics',
        'callback': 'callback',
        '': {
            'callbacks': {
                'Overview': 'callbacks',
                'HookCallback': 'callbacks.hooks',
                'MixedPrecision': 'callbacks.fp16',
                'OneCycleScheduler': 'callbacks.one_cycle',
                'LRFinder': 'callbacks.lr_finder',
                'MixUpCallback': 'callbacks.mixup',
                'RNNTrainer': 'callbacks.rnn',
                'GeneralScheduler': 'callbacks.general_sched',
                'Tracking callbacks': 'callbacks.tracking'
            }
        },
    },
    'Applications': {
        'Overview': 'applications',
        '': {
            'vision': {
                'Overview': 'vision',
                'vision.learner': 'vision.learner',
                'vision.transform': 'vision.transform',
                'vision.image': 'vision.image',
                'vision.data': 'vision.data',
                'vision.model overview': 'vision.models',
                'vision.models.unet': 'vision.models.unet'
            }
        },
        'empty1': {
            'text': {
                'Overview': 'text',
                'text.learner': 'text.learner',
                'text.transform': 'text.transform',
                'text.data': 'text.data',
                'text.models': 'text.models'
            },
        },
        'empty2': {
            'tabular': {
                'Overview': 'tabular',
                'tabular.transform': 'tabular.transform',
                'tabular.data': 'tabular.data',
                'tabular.models': 'tabular.models'
            },
        },
        'collab': 'collab',
    },
    'Core': {
        'Overview': 'overview',
        'data_block': 'data_block',
        'basic_data': 'basic_data',
        'layers': 'layers',
        'datasets': 'datasets',
        'core': 'core',
        'torch_core': 'torch_core',
    },
    'Doc authoring': {
        'Overview': 'gen_doc',
        'gen_doc.gen_notebooks': 'gen_doc.gen_notebooks',
        'gen_doc.nbdoc': 'gen_doc.nbdoc',
        'gen_doc.convert2html': 'gen_doc.convert2html',
    },
    'Library development': {
        'Overview': 'https://docs-dev.fast.ai/develop',
        'Testing': 'https://docs-dev.fast.ai/test',
        'Style guide': 'https://docs-dev.fast.ai/style',
        'Abbreviation': 'https://docs-dev.fast.ai/abbr',
        'Typing': 'fastai_typing',
        'Packaging fastai': 'https://docs-dev.fast.ai/release'
    }
}
