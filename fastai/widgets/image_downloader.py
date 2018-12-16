from ..core import *
from ipywidgets import widgets, Layout, Output, HBox, VBox, Text, BoundedIntText, Button, Dropdown, Box
from IPython.display import clear_output, display
from google_images_download import google_images_download

__all__ = ['ImageDownloader']

class ImageDownloader():
    """
    Displays a simple widget that allows searching and downloading images from google images search
    in a Jupyter notebook.
    """

    def __init__(self, path:Union[Path,str]='data'):
        "Setup path to save images to, init the UI, and render the widgets."
        self._path = Path(path)
        self._ui = self._init_ui()
        self.render()

    def _init_ui(self) -> VBox:
        """
        Initialize the widget UI and return the UI, 
        only used to initially create all the components and setup event handlers."
        """
        self._search_input = Text(placeholder="What images to search for?")
        self._count_input = BoundedIntText(placeholder="How many pics?", 
                                        value=10, min=1, max=100, step=1, 
                                        layout=Layout(width='60px'))
        self._size_input = Dropdown(options= ['>400*300', '>640*480', '>800*600', '>1024*768', '>2MP', '>4MP', '>6MP', '>8MP', '>10MP'],
                                    value='>400*300',
                                    layout=Layout(width='120px'))
        self._download_button = Button(description="Search & Download", icon="download", layout=Layout(width='200px'))
        self._download_button.on_click(self.on_download_button_click) 

        self._output = Output()

        # Top horizontal controls bar
        self._controls_pane  = HBox([self._search_input, 
                                    self._count_input,
                                    self._size_input,
                                    self._download_button],
                                    layout=Layout(width='auto', height='40px'))

        self._heading = ""
        self._download_complete_heading = "<h3>Download complete. Here are a few images</h3>"
        self._preview_header = widgets.HTML(self._heading, layout=Layout(height='60px'))
        self._img_pane = Box(layout=Layout(display='inline'))
        return VBox([self._controls_pane, self._preview_header, self._img_pane])


    def render(self) -> None:
        "Render the image search widget."
        clear_output()
        display(self._ui)

    def clear_imgs(self) -> None:
        "Clear the widget's images preview pane."
        self._preview_header.value = self._heading
        self._img_pane.children = tuple()

    def validate_search_input(self) -> bool:
        "Check if input value is empty."
        input = self._search_input
        if input.value == str():
            input.layout = Layout(border="solid 2px red", height='auto')
        else:
            self._search_input.layout = Layout()
        return input.value != str()

    def on_download_button_click(self, btn) -> None:
        "Download button click handler: validate search term and download images."
        term = self._search_input.value           
        limit = int(self._count_input.value)
        size = self._size_input.value

        if not self.validate_search_input(): return
        
        self.clear_imgs() 
        self.download_images(term, limit=limit, size=size)
        self.display_images_widgets(self.get_preview_images_fnames(term)[:min(limit, 12)])
        self._preview_header.value = self._download_complete_heading
        self.render()
        
    def display_images_widgets(self, fnames:list) -> None:
        "Display a few preview images in the notebook"
        imgs = [widgets.Image(value=open(f, 'rb').read(), width='200px') for f in fnames]
        self._img_pane.children = tuple(imgs)

    def get_preview_images_fnames(self, keyword:str) -> list:  
        "Fetch all the image file paths in for a given keyword"
        image_file_formats = ['.jpg', '.jpeg', '.png']
        return [i for i in (self._path/keyword).iterdir()
            if i.is_file() and i.suffix in image_file_formats]
        
    def download_images(self, search_term:str, path:Union[Path,str]=None, 
                        limit:int=10, size:str='>400*300') -> None:
        "Download up to `limit` images from google search with `search_term`."
        if path is None: path = self._path
        dl = google_images_download.googleimagesdownload()
        dl.download({
            'limit':limit,
            'output_directory': str(path),
            'keywords':search_term, 
            'size':size })
        

