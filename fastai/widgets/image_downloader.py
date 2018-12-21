from ..core import *
from ipywidgets import widgets, Layout, Output, HBox, VBox, Text, BoundedIntText, Button, Dropdown, Box
from IPython.display import clear_output, display
from urllib.parse import quote
from bs4 import BeautifulSoup

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
        google_images_parse_and_download(path, search_term, size=size, n_images=limit)


def google_images_parse_and_download(path:PathOrStr, search_term:str, size:str='>400*300', n_images:int=10, format:str='jpg', max_workers:int=8) -> None:
    "Search Google for images that match the `search_term` and `size`, up to `n_images` pics. Then download them into `path` subfolder."
    search_url = google_images_search_url(search_term, size, format)
    img_tuples = google_images_parse_urls(search_url, format=format, n_images=n_images)
    google_images_download(path, search_term, img_tuples, max_workers=max_workers)

def google_images_url_params(size:str='>400*300', format:str='jpg') -> str:
    "Build Google Images Search Url params and return them as a string."
    size_options = {'large':'isz:l','medium':'isz:m','icon':'isz:i','>400*300':'isz:lt,islt:qsvga','>640*480':'isz:lt,islt:vga','>800*600':'isz:lt,islt:svga','>1024*768':'visz:lt,islt:xga','>2MP':'isz:lt,islt:2mp','>4MP':'isz:lt,islt:4mp','>6MP':'isz:lt,islt:6mp','>8MP':'isz:lt,islt:8mp','>10MP':'isz:lt,islt:10mp','>12MP':'isz:lt,islt:12mp','>15MP':'isz:lt,islt:15mp','>20MP':'isz:lt,islt:20mp','>40MP':'isz:lt,islt:40mp','>70MP':'isz:lt,islt:70mp'}
    format_options = {'jpg':'ift:jpg','gif':'ift:gif','png':'ift:png','bmp':'ift:bmp','svg':'ift:svg','webp':'webp','ico':'ift:ico'}
    return "&tbs=" + size_options[size] + "," + format_options[format]

def google_images_search_url(search_term:str, size:str='>400*300', format:str='jpg') -> str:
    "Return a Google Images Search URL for a given search term."
    return ( 'https://www.google.com/search?q=' +
            quote(search_term) +
            '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' +
            google_images_url_params(size, format) +
            '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg' )

def img_fname(img_url:str) -> str:
    "Return image file name including the extension given it's url."
    return img_url.split('/')[-1]

def google_images_parse_urls(url:str, format:str='jpg', n_images:int=10) -> list:
    "Parse the Google Images Search for urls and return the image metadata as tuples (fname, url)."
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}
    html = requests.get(url, headers=headers).text
    bs = BeautifulSoup(html, 'html.parser')
    img_tags = bs.find_all('div', {'class': 'rg_meta'})
    metadata_dicts = (json.loads(e.text) for e in img_tags)
    img_tuples = ((img_fname(d['ou']), d['ou']) for d in metadata_dicts if d['ity'] == format)
    return list(itertools.islice(img_tuples, n_images))

def google_images_download(path:PathOrStr, label:str, img_tuples:list, max_workers:int=8) -> None:
    "Downloads images to `path`/`label`."
    os.makedirs(Path(path)/label, exist_ok=True)
    parallel( partial(_download_single_google_image, Path(path)/label), img_tuples, max_workers=max_workers)

def _download_single_google_image(dest_folder:Path, img_tuple:tuple, i:int) -> None:
    "Downloads a single image from Google Search results to `dest_folder`."
    suffix = re.findall(r'\.\w+?(?=(?:\?|$))', img_tuple[1])
    suffix = suffix[0] if len(suffix)>0  else '.jpg'
    fname = f"{i:08d}{suffix}"
    download_url(img_tuple[1], dest_folder/fname)

