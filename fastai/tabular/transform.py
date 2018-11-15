"Cleaning and feature engineering functions for structured data"
from ..torch_core import *

__all__ = ['Categorify', 'FillMissing', 'FillStrategy', 'Normalize', 'TabularProc']

@dataclass
class TabularProc():
    "A transform for tabular dataframes."
    cat_names:StrList
    cont_names:StrList

    def __call__(self, df:DataFrame, test:bool=False):
        "Apply the correct function to `df` depending on `test`."
        func = self.apply_test if test else self.apply_train
        func(df)

    def apply_train(self, df:DataFrame):
        "Function applied to `df` if it's the train set."
        raise NotImplementedError
    def apply_test(self, df:DataFrame):
        "Function applied to `df` if it's the test set."
        self.apply_train(df)

class Categorify(TabularProc):
    "Transform the categorical variables to that type."

    def apply_train(self, df:DataFrame):
        self.categories = {}
        for n in self.cat_names:
            df.loc[:,n] = df.loc[:,n].astype('category').cat.as_ordered()
            self.categories[n] = df[n].cat.categories

    def apply_test(self, df:DataFrame):
        for n in self.cat_names:
            df.loc[:,n] = pd.Categorical(df[n], categories=self.categories[n], ordered=True)

FillStrategy = IntEnum('FillStrategy', 'MEDIAN COMMON CONSTANT')

@dataclass
class FillMissing(TabularProc):
    "Fill the missing values in continuous columns."
    fill_strategy:FillStrategy=FillStrategy.MEDIAN
    add_col:bool=True
    fill_val:float=0.

    def apply_train(self, df:DataFrame):
        self.na_dict = {}
        for name in self.cont_names:
            if pd.isnull(df.loc[:,name]).sum():
                if self.add_col:
                    df.loc[:,name+'_na'] = pd.isnull(df.loc[:,name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                if self.fill_strategy == FillStrategy.MEDIAN: filler = df.loc[:,name].median()
                elif self.fill_strategy == FillStrategy.CONSTANT: filler = self.fill_val
                else: filler = df.loc[:,name].dropna().value_counts().idxmax()
                df.loc[:,name] = df.loc[:,name].fillna(filler)
                self.na_dict[name] = filler

    def apply_test(self, df:DataFrame):
        for name in self.cont_names:
            if name in self.na_dict:
                if self.add_col:
                    df.loc[:,name+'_na'] = pd.isnull(df[name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                df.loc[:,name] = df.loc[:,name].fillna(self.na_dict[name])

class Normalize(TabularProc):
    "Transform the categorical variables to that type."

    def apply_train(self, df:DataFrame):
        self.means,self.stds = {},{}
        for n in self.cont_names:
            self.means[n],self.stds[n] = df.loc[:,n].mean(),df.loc[:,n].std()
            df.loc[:,n] = (df.loc[:,n]-self.means[n]) / (1e-7 + self.stds[n])

    def apply_test(self, df:DataFrame):
        for n in self.cont_names:
            df.loc[:,n] = (df.loc[:,n]-self.means[n]) / (1e-7 + self.stds[n])
