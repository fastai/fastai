from .imports import *
from .torch_imports import *
from .core import *

def add_datepart(df, date_fld, yr_fld="Year", month_fld="Month", week_fld="Week", day_fld="Day"):
    df[date_fld] = pd.to_datetime(df[date_fld])
    df[yr_fld] = df.Date.dt.year
    df[month_fld] = df.Date.dt.month
    df[week_fld] = df.Date.dt.week
    df[day_fld] = df.Date.dt.day

def combine_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)

