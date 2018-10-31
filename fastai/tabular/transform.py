"Cleaning and feature engineering functions for structured data"
from ..torch_core import *

__all__ = ['Categorify', 'FillMissing', 'FillStrategy', 'TabularTransform',
           'RemoveMinVariance']

@dataclass
class TabularTransform():
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

class Categorify(TabularTransform):
    "Transform the categorical variables to that type."

    def apply_train(self, df:DataFrame):
        self.categories = {}
        for n in self.cat_names:
            df[n] = df[n].astype('category').cat.as_ordered()
            self.categories[n] = df[n].cat.categories

    def apply_test(self, df:DataFrame):
        for n in self.cat_names:
            df[n] = pd.Categorical(df[n], categories=self.categories[n], ordered=True)

FillStrategy = IntEnum('FillStrategy', 'MEDIAN COMMON CONSTANT')

@dataclass
class FillMissing(TabularTransform):
    "Fill the missing values in continuous columns."
    fill_strategy:FillStrategy=FillStrategy.MEDIAN
    add_col:bool=True
    fill_val:float=0.

    def apply_train(self, df:DataFrame):
        self.na_dict = {}
        for name in self.cont_names:
            if pd.isnull(df[name]).sum():
                if self.add_col:
                    df[name+'_na'] = pd.isnull(df[name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                if self.fill_strategy == FillStrategy.MEDIAN: filler = df[name].median()
                elif self.fill_strategy == FillStrategy.CONSTANT: filler = self.fill_val
                else: filler = df[name].dropna().value_counts().idxmax()
                df[name] = df[name].fillna(filler)
                self.na_dict[name] = filler

    def apply_test(self, df:DataFrame):
        for name in self.cont_names:
            if name in self.na_dict:
                if self.add_col:
                    df[name+'_na'] = pd.isnull(df[name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                df[name] = df[name].fillna(self.na_dict[name])

@dataclass
class RemoveMinVariance(TabularTransform):
    "Remove variables below a certain variance threshold."
    min_variance:float=0.00001
    remove_cols:bool=True
    check_sample:float=1.0
    verbose:bool=True

    def apply_train(self, df:DataFrame):
        self.columns_to_drop = []
        for n in self.cont_names:
            current_column = df[n]
            if self.check_sample < 1.0:
                n_variance = (current_column
                              .sample(int(len(current_column) * check_sample))
                              .var())
            else:
                n_variance = current_column.var()
            if n_variance < self.min_variance:
                if self.verbose:
                    if self.remove_cols:
                        print(("Dropping column '%s' since its variance of %0.4f"
                               + " is below the threshold") % (n, n_variance))
                    else:
                        print(("Attention: The variance of column '%s' is %0.4f"
                               + " and thus below the threshold") % (n, n_variance))
                if self.remove_cols:
                    self.columns_to_drop.append(n)

        df.drop(columns=self.columns_to_drop, inplace=True)

    def apply_test(self, df:DataFrame):
        df.drop(columns=self.columns_to_drop, inplace=True)
