# Usage: After editing this file, next run:
#
# tools/make_sidebar.py
# git commit docs/_data/sidebars/home_sidebar.yml docs_src/sidebar/sidebar_data.py
# git push

# This dict defines the structure:

sidebar_d = {
    'Getting started': {
        'Installation': 'https://github.com/fastai/fastai/blob/master/README.md#installation',
        'Installation Extras': '/install',
        'Troubleshooting': '/troubleshoot',
        'Performance': '/performance',
        'Support': '/support'
    },
    'Training': {
        'Overview': '/training',
        'basic_train': '/basic_train',
        'train': '/train',
        'metrics': '/metrics',
        'callback': '/callback',
        '': {
            'callbacks': {
                'Overview': '/callbacks',
                'HookCallback': '/callbacks.hooks',
                'MixedPrecision': '/callbacks.fp16',
                'OneCycleScheduler': '/callbacks.one_cycle',
                'LRFinder': '/callbacks.lr_finder',
                'MixUpCallback': '/callbacks.mixup',
                'RNNTrainer': '/callbacks.rnn',
                'GeneralScheduler': '/callbacks.general_sched',
                'CSV Logger': '/callbacks.csv_logger',
                'Tracking': '/callbacks.tracker',
                'Tensorboard': '/callbacks.tensorboard',
                'Memory Profiling': '/callbacks.mem',
                'Miscellaneous': '/callbacks.misc',
            }
        },
    },
    'Applications': {
        'Overview': '/applications',
        '': {
            'vision': {
                'Overview': '/vision',
                'vision.learner': '/vision.learner',
                'vision.interpret': '/vision.interpret',
                'vision.transform': '/vision.transform',
                'vision.image': '/vision.image',
                'vision.data': '/vision.data',
                'vision.gan': '/vision.gan',
                'vision.model overview': '/vision.models',
                'vision.models.unet': '/vision.models.unet',
            }
        },
        'empty1': {
            'text': {
                'Overview': '/text',
                'text.learner': '/text.learner',
                'text.interpret': '/text.interpret',
                'text.transform': '/text.transform',
                'text.data': '/text.data',
                'text.models': '/text.models'
            },
        },
        'empty2': {
            'tabular': {
                'Overview': '/tabular',
                'tabular.transform': '/tabular.transform',
                'tabular.data': '/tabular.data',
                'tabular.models': '/tabular.models',
                'tabular.learner': '/tabular.learner'
            },
        },
        'empty3': {
            'widgets': {
                'widgets.class_confusion': '/widgets.class_confusion',
                'widgets.image_cleaner': '/widgets.image_cleaner'
            },
        },
        'collab': '/collab',
    },
    'Core': {
        'Overview': '/overview',
        'data_block': '/data_block',
        'basic_data': '/basic_data',
        'layers': '/layers',
        'datasets': '/datasets',
        'core': '/core',
        'torch_core': '/torch_core',
        'imports': '/imports',
    },
    'Utils': {
        'Helpers': '/utils.collect_env',
        'Memory Management': '/utils.mem',
        'ipython helpers': '/utils.ipython',
        'Display utils': '/utils.mod_display',
    },
    'Tutorials': {
        'Overview': '/tutorials',
        'Look at data': '/tutorial.data',
        'Inference Learner': '/tutorial.inference',
        'Custom ItemList': '/tutorial.itemlist',
        'DL on a Shoestring': '/tutorial.resources',
        'Distributed training': '/distributed',
    },
    'Doc authoring': {
        'Instructions': '/gen_doc_main',
        'gen_doc': '/gen_doc',
        'gen_doc.gen_notebooks': '/gen_doc.gen_notebooks',
        'gen_doc.nbdoc': '/gen_doc.nbdoc',
        'gen_doc.nbtest': '/gen_doc.nbtest',
        'gen_doc.convert2html': '/gen_doc.convert2html',
    },
    'Library development': {
        'Contributing': 'https://github.com/fastai/fastai/blob/master/CONTRIBUTING.md',
        'Dev Notes': '/dev/develop',
        'GPU Notes': '/dev/gpu',
        'git notes': '/dev/git',
        'Testing': '/dev/test',
        'Style Guide': '/dev/style',
        'Abbreviations': '/dev/abbr',
        'Packaging': '/dev/release',
    }
}
