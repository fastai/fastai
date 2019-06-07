from ..fastai.vision.data import COCO_download, COCO_load, COCODataset, LoadVideo

import mock
import unittest


class COCOTestCase(unittest.TestCase):
    annots = {}
    @mock.patch('mymodule.os')
    @mock.patch('..fastai.vision.data.urllib')

    def COCO_download_test(self, mock_os):
        COCO_download()
        # test that rm called os.remove with the right parameters
        #mock_os.remove.assert_called_with("any path")