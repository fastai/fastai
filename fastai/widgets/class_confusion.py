import math
import pandas as pd
import matplotlib.pyplot as plt

from itertools import permutations, combinations
from fastai.train import ClassificationInterpretation
import ipywidgets as widgets

class ClassLosses():
    
    """Plot the most confused datapoints and statistics for your misses. 
    \nPass in a `interp` object and a list of classes to look at. 
    Optionally you can include an odered list in the form of [[class_1, class_2]],
    \n a figure size, and a cut_off limit for the maximum categorical categories to use on a variable"""
    def __init__(self, interp:ClassificationInterpretation, classlist:list, 
               is_ordered:bool=False, cut_off:int=100, varlist:list=None,
               figsize:tuple=(8,8)):
        self.interp = interp
        if str(type(interp.learn.data)) == "<class 'fastai.tabular.data.TabularDataBunch'>":
            if interp.learn.data.train_ds.x.cont_names != []: 
                for x in range(len(interp.learn.data.procs)):
                      if "Normalize" in str(interp.learn.data.procs[x]):
                            self.means = interp.learn.data.train_ds.x.processor[0].procs[x].means
                            self.stds = interp.learn.data.train_ds.x.processor[0].procs[x].stds
        self.is_ordered = is_ordered
        self.cut_off = cut_off
        self.figsize = figsize
        self.vars = varlist
        self.show_losses(classlist)
    

    
    def create_tabs(self, df_list:list, cat_names:list):
        self.cols = math.ceil(math.sqrt(len(df_list)))
        self.rows = math.ceil(len(df_list)/self.cols)
        self.boxes = len(df_list)
        df_list[0].columns = df_list[0].columns.get_level_values(0)
        tbnames = list(df_list[0].columns)
        tbnames = tbnames[:-1]
        items = [widgets.Output() for i, tab in enumerate(tbnames)]
        self.tbnames = tbnames
        self.tabs = widgets.Tab()
        self.tabs.children = items
        for i in range(len(items)):
            self.tabs.set_title(i, tbnames[i])
        self.populate_tabs(self.classl)
      
    def populate_tabs(self, classl:list):
        for i, tab in enumerate(self.tbnames):
            with self.tabs.children[i]:
                if self.boxes is not None:
                    fig, ax = plt.subplots(self.boxes, figsize=self.figsize)
                    fig.subplots_adjust(hspace=.5)
                else:
                    fig, ax = plt.subplots(self.cols, self.rows, figsize=self.figsize)
                    fig.subplots_adjust(hspace=.5)
                for j, x in enumerate(self.dfs):
                    if self.boxes is None:
                        row = int(j / self.cols)
                        col = j % row
                    if tab in self.cat_names:
                        vals = pd.value_counts(x[tab].values)
                        ttl = str.join('', x.columns[-1])
                        if j == 0:
                            title = ttl + ' ' + tab + ' distribution'
                        else:
                            title = 'Misclassified ' + ttl + ' ' + tab + ' distribution'
                        if self.boxes is not None:
                            if vals.nunique() < 10:
                                fig = vals.plot(kind='bar', title=title,  ax=ax[j], rot=0, width=.75)
                            else:
                                fig = vals.plot(kind='barh', title=title,  ax=ax[j], width=.75)   
                        else:
                            fig = vals.plot(kind='barh', title=title,  ax=ax[row, col], width=.75)
                    else:
                        vals = x[tab]
                        ttl = str.join('', x.columns[-1])
                        if j == 0:
                            title = ttl + ' ' + tab + ' distribution'
                        else:
                            title = 'Misclassified ' + ttl + ' ' + tab + ' distrobution'
                        if self.boxes is not None:
                            axs = vals.plot(kind='hist', ax=ax[j], title=title, y='Frequency')
                        else:
                            axs = vals.plot(kind='hist', ax=ax[row, col], title=title, y='Frequency')
                        axs.set_ylabel('Frequency')
                        if len(set(vals)) > 1:
                            vals.plot(kind='kde', ax=axs, title=title, secondary_y=True)
                        else:
                            print('Less than two unique values, cannot graph the KDE')
                plt.show(fig)
                plt.tight_layout
        display(self.tabs)
        
    def show_losses(self, classl:list, **kwargs):
        if str(type(self.interp.learn.data)) == "<class 'fastai.tabular.data.TabularDataBunch'>":
            self.tab_losses(classl)
        else:
            self.im_losses(classl)

    def im_losses(self, classl:list, **kwargs):
        if self.is_ordered == True:
            lis = classl
        else: 
            lis = list(permutations(classl, 2))
        self.tl_val, self.tl_idx = self.interp.top_losses(len(self.interp.losses))
        classes_gnd = self.interp.data.classes
        vals = self.interp.most_confused()
        ranges = []
        tbnames = []
        k = input('Please enter a value for `k`, or the top images you will see: ')
        k = int(k)
        self.k = k
        for x in iter(vals):
            for y in range(len(lis)):
                if x[0:2] == lis[y]:
                    ranges.append(x[2])
                    tbnames.append(str(x[0] + ' | ' + x[1]))
        items = [widgets.Output() for i, tab in enumerate(tbnames)]
        self.tbnames = tbnames
        self.tabs = widgets.Tab()
        self.tabs.children = items
        for i in range(len(items)):
            self.tabs.set_title(i, tbnames[i])
        self.ranges = ranges
        self.classl = classl
        for i, tab in enumerate(self.tbnames):
            with self.tabs.children[i]:
                x = 0
                if self.ranges[i] < k:
                    cols = math.ceil(math.sqrt(self.k))
                    rows = math.ceil(self.ranges[i]/cols)
                    fig, ax = plt.subplots(rows, cols, figsize=self.figsize)

                if self.ranges[i] < 4:
                    cols = 2
                    rows = 2
                    fig, ax = plt.subplots(rows, cols, figsize=self.figsize)
                else:
                    cols = math.ceil(math.sqrt(self.k))
                    rows = math.ceil(self.k/cols)
                    fig, ax = plt.subplots(rows, cols, figsize=self.figsize)

                [axi.set_axis_off() for axi in ax.ravel()]
                for j, idx in enumerate(self.tl_idx):
                    if k < x+1 or x > self.ranges[i]:
                        break
                    da, cl = self.interp.data.dl(self.interp.ds_type).dataset[idx]
                    row = (int)(x / cols)
                    col = x % cols

                    ix = int(cl)
                    if str(cl) == tab.split(' ')[0] and str(classes_gnd[self.interp.pred_class[idx]]) == tab.split(' ')[2]:
                        img, lbl = self.interp.data.valid_ds[idx]
                        fn = self.interp.data.valid_ds.x.items[idx]
                        fn = re.search('([^/*]+)_\d+.*$', str(fn)).group(0)
                        img.show(ax=ax[row, col])
                        ax[row,col].set_title(fn)
                        x += 1
                plt.show(fig)
                plt.tight_layout()
        display(self.tabs)

    def tab_losses(self, classl:list, **kwargs):
        tl_val, tl_idx = self.interp.top_losses(len(self.interp.losses))
        classes = self.interp.data.classes
        cat_names = self.interp.data.x.cat_names
        cont_names = self.interp.data.x.cont_names
        if self.is_ordered == False:
            comb = list(permutations(classl,2))
        else:
            comb = classl

        dfarr = []

        arr = []
        for i, idx in enumerate(tl_idx):
            da, _ = self.interp.data.dl(self.interp.ds_type).dataset[idx]
            res = ''
            for c, n in zip(da.cats, da.names[:len(da.cats)]):
                string = f'{da.classes[n][c]}'
                if string == 'True' or string == 'False':
                    string += ';'
                    res += string

                else:
                    string = string[1:]
                    res += string + ';'
            for c, n in zip(da.conts, da.names[len(da.cats):]):
                res += f'{c:.4f};'
            arr.append(res)
        f = pd.DataFrame([ x.split(';')[:-1] for x in arr], columns=da.names)
        for i, var in enumerate(self.interp.data.cont_names):
            f[var] = f[var].apply(lambda x: float(x) * self.stds[var] + self.means[var])
        f['Original'] = 'Original'
        dfarr.append(f)


        for j, x in enumerate(comb):
            arr = []
            for i, idx in enumerate(tl_idx):
                da, cl = self.interp.data.dl(self.interp.ds_type).dataset[idx]
                cl = int(cl)

                if classes[self.interp.pred_class[idx]] == comb[j][0] and classes[cl] == comb[j][1]:
                    res = ''
                    for c, n in zip(da.cats, da.names[:len(da.cats)]):
                        string = f'{da.classes[n][c]}'
                        if string == 'True' or string == 'False':
                            string += ';'
                            res += string
                        else:
                            string = string[1:]
                            res += string + ';'
                    for c, n in zip(da.conts, da.names[len(da.cats):]):
                        res += f'{c:.4f};'
                    arr.append(res)      
            f = pd.DataFrame([ x.split(';')[:-1] for x in arr], columns=da.names)
            for i, var in enumerate(self.interp.data.cont_names):
                f[var] = f[var].apply(lambda x: float(x) * self.stds[var] + self.means[var])
            f[str(x)] = str(x)
            dfarr.append(f)
        self.dfs = dfarr
        self.cat_names = cat_names
        self.classl = classl
        self.create_tabs(dfarr, cat_names)