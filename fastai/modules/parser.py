import argparse

def parser_arguments():
    '''
    Manage the input from the terminal.
    :return: parser
    '''
    parser = argparse.ArgumentParser(description='Open Image Dataset Downloader')

    parser.add_argument("command",
                        metavar="<command> 'downloader', 'visualizer' or 'ill_downloader'.",
                        help="'downloader', 'visualizer' or 'ill_downloader'.")
    parser.add_argument('--Dataset', required=False,
                        metavar="/path/to/OID/csv/",
                        help='Directory of the OID dataset folder')
    parser.add_argument('--classes', required=False, nargs='+',
                        metavar="list of classes",
                        help="Sequence of 'strings' of the wanted classes")
    parser.add_argument('--type_csv', required=False, choices=['train', 'test', 'validation', 'all'],
                        metavar="'train' or 'validation' or 'test' or 'all'",
                        help='From what csv search the images')

    parser.add_argument('--sub', required=False, choices=['h', 'm'], 
                        metavar="Subset of human verified images or machine generated (h or m)",
                        help='Download from the human verified dataset or from the machine generated one.')

    parser.add_argument('--image_IsOccluded', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object is occluded by another object in the image.')
    parser.add_argument('--image_IsTruncated', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object extends beyond the boundary of the image.')
    parser.add_argument('--image_IsGroupOf', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the box spans a group of objects (min 5).')
    parser.add_argument('--image_IsDepiction', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object is a depiction.')
    parser.add_argument('--image_IsInside', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates a picture taken from the inside of the object.')

    parser.add_argument('--multiclasses', required=False, default='0', choices=['0', '1'],
                       metavar="0 (default) or 1",
                       help='Download different classes separately (0) or together (1)')

    parser.add_argument('--n_threads', required=False, metavar="[default 20]",
                       help='Num of the threads to use')

    parser.add_argument('--noLabels', required=False, action='store_true',
                       help='No labels creations')

    parser.add_argument('--limit', required=False, type=int, default=None,
                        metavar="integer number",
                        help='Optional limit on number of images to download')
    
    return parser.parse_args()
