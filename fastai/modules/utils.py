import os
from textwrap import dedent

def images_options(df_val, args):
    '''
    Manage the options for the images downloader.

    :param df_val: DataFrame Value.
    :param args: argument parser.
    :return: modified df_val
    '''
    if args.image_IsOccluded is not None:
        rejectedID = df_val.ImageID[df_val.IsOccluded != int(args.image_IsOccluded)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    if args.image_IsTruncated is not None:
        rejectedID = df_val.ImageID[df_val.IsTruncated != int(args.image_IsTruncated)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    if args.image_IsGroupOf is not None:
        rejectedID = df_val.ImageID[df_val.IsGroupOf != int(args.image_IsGroupOf)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    if args.image_IsDepiction is not None:
        rejectedID = df_val.ImageID[df_val.IsDepiction != int(args.image_IsDepiction)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    if args.image_IsInside is not None:
        rejectedID = df_val.ImageID[df_val.IsInside != int(args.image_IsInside)].values
        df_val = df_val[~df_val.ImageID.isin(rejectedID)]

    return df_val

def mkdirs(Dataset_folder, csv_folder, classes, type_csv):
    '''
    Make the folder structure for the system.

    :param Dataset_folder: Self explanatory
    :param csv_folder: folder path of csv files
    :param classes: list of classes to download
    :param type_csv: train, validation, test or all 
    :return: None
    '''

    directory_list = ['train', 'validation', 'test']
    
    if not type_csv == 'all':
        for class_name in classes:
            if not Dataset_folder.endswith('_nl'):
                folder = os.path.join(Dataset_folder, type_csv, class_name, 'Label')
            else:
                folder = os.path.join(Dataset_folder, type_csv, class_name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            filelist = [f for f in os.listdir(folder) if f.endswith(".txt")]
            for f in filelist:
                os.remove(os.path.join(folder, f))

    else:
        for directory in directory_list:
            for class_name in classes:
                if not Dataset_folder.endswith('_nl'):
                    folder = os.path.join(Dataset_folder, directory, class_name, 'Label')
                else:
                    folder = os.path.join(Dataset_folder, directory, class_name, 'Label')
                if not os.path.exists(folder):
                    os.makedirs(folder)
                filelist = [f for f in os.listdir(folder) if f.endswith(".txt")]
                for f in filelist:
                    os.remove(os.path.join(folder, f))

    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

def progression_bar(total_images, index):
    '''
    Print the progression bar for the download of the images.

    :param total_images: self explanatory
    :param index: self explanatory
    :return: None
    '''
    # for windows os
    if os.name == 'nt':
        from ctypes import windll, create_string_buffer

        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)

        if res:
            import struct
            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            columns = right - left + 1
            rows = bottom - top + 1
        else:
            columns, rows = 80, 25 # can't determine actual size - return default values
    # for linux/gnu os
    else:
        rows, columns = os.popen('stty size', 'r').read().split()
    toolbar_width = int(columns) - 10
    image_index = index
    index = int(index / total_images * toolbar_width)

    print(' ' * (toolbar_width), end='\r')
    bar = "[{}{}] {}/{}".format('-' * index, ' ' * (toolbar_width - index), image_index, total_images)
    print(bar.rjust(int(columns)), end='\r')

def show_classes(classes):
    '''imag
    Show the downloaded classes in the selected folder during visualization mode
    '''
    for n in classes:
        print("- {}".format(n))
    print("\n")

def logo(command):
    return None

class bcolors:
    HEADER = '\033[95m'
    
    INFO = '    [INFO] | '
    OKBLUE = '\033[94m[DOWNLOAD] | '
    WARNING = '\033[93m    [WARN] | '
    FAIL = '\033[91m   [ERROR] | '

    OKGREEN = '\033[92m'
    ENDC = '\033[0m'