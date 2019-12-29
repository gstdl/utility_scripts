import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from time import time
from matplotlib import rc,rcParams

def set_notebook(style='seaborn-darkgrid',col_row=99999999999, figsize=(15,5)):
    global fsize,maxrows,maxcols,style_
    fsize = rcParams['figure.figsize'] = figsize
    maxrows = pd.options.display.max_rows = col_row
    maxcols = pd.options.display.max_columns = col_row
    style_ = plt.style.use(style)

def memory_usage(df):
    return '{:.4f} MB'.format(df.memory_usage().sum()/1024**2)

def resumetable(df):
    summary = pd.DataFrame(dict(dataFeatures=df.columns,
                            dataType=df.dtypes,
                            null=df.isna().sum(),
                            nullPct=round(df.isna().sum()/len(df)*100,2),
                            unique=df.nunique(),
#                              uniqueSample=[list(df[i].drop_duplicates().sample(2)) for i in df.columns]
                        )).set_index('dataFeatures')


    for feature in summary.index:
        summary.loc[feature,'entropy'] = entropy(df[feature].value_counts(normalize=True),base=2)

        for s in ['mostValue','leastValue']:
            for i in range(3):
                if s=='leastValue':
                    j=-(1+i)
                else:
                    j=i
                try:
                    idx=df[feature].dropna().value_counts().index[j]
                    summary.loc[feature,f'{s}_{i+1}']=idx
                    summary.loc[feature,f'{s}Count_{i+1}']='{} ({:.2f}%)'.format(df[feature].dropna().value_counts()[idx],df[feature].dropna().value_counts(normalize=True)[idx]*100)
                except:
                    summary.loc[feature,f'{s}_{i+1}']='-'
                    summary.loc[feature,f'{s}Count_{i+1}']='-'
    print('Dataframe shape = ',df.shape)
    return summary

def reduce_mem_usage(df, verbose=True):
    global data
    data = df
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased from {:5.2f} to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

def to_datetime(df,cols):
    if type(cols) != list:
        if type(cols) in [str,int,float]:
            cols = [cols]
        else:
            cols = list(cols)
    not_found = []
    for col in cols:
        if col not in df.columns:
            not_found.append(col)
    if len(not_found)>0:
        raise AttributeError(not_found, 'is not found in', list(df.columns))
    global data
    data = df
    print('___Before Conversion___')
    print(df[cols].info())
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    print('___After Conversion___')
    print(df[cols].info())

def get_timedelta(df, cols, new_col = 'time_delta', order = 0, preview=10):
    if len(cols)!=2:
        raise AttributeError('Function only operate on 2 columns')
    if type(cols) not in [list,tuple,set]:
        raise AttributeError('cols attribute only accepts list, tuples, or sets')
    cols = list(cols)
    try:
        ser=(df[cols[0]]-df[cols[1]]).dt.days
    except:
        for col in cols:
            df[col] = pd.to_datetime(df[col])
        ser=(df[cols[0]]-df[cols[1]]).dt.days
    global data
    data = df
    if order == 0:
        df[new_col] = ser
    elif order ==1:
        df[new_col] = ser*-1
    return df.head(preview)

def viz_catcol(df,cols,top=10,bottom=10, hue=None,ax_grid0=True,ax_grid1=False,figsize=(15,5)):
    import warnings
    warnings.filterwarnings('ignore')
    if hue!=None:
        hue=df[hue]
    if type(cols) != list:
        if type(cols) in [str,int,float]:
            cols = [cols]
        else:
            cols = list(cols)
    not_found = []
    for col in cols:
        if col not in df.columns:
            not_found.append(col)
    if len(not_found)>0:
        raise AttributeError(not_found, 'is not found in', list(df.columns))
    rows = len(cols)
    figsize=(figsize[0],figsize[1]*rows)
    if top!=0 and bottom!=0:
        columns=2
    else:
        columns=1
    fig,axes=plt.subplots(rows,columns,figsize=figsize)
    for col, axe in zip(cols,axes):
        df.loc[:,col]=df.loc[:,col].copy().fillna('NaN')
        df[col].replace({i:'\n'.join([f"{' '.join(i.split()[j:j+5])}" for j in range(0,len(i.split()),5)]) for i in df[col].unique() if type(i)==str},inplace=True)
        ser = df[col].value_counts(normalize=False)
        ln = len(ser)
        if ln>top:
            top_=top
        else:
            top_=ln
        if ln>bottom:
            bottom_=bottom
        else:
            bottom_=ln
        if top+bottom==0:
            sns.countplot(y=df[col],ax=axe,order=ser.index,hue=hue)
            ax1=axe.twiny()
            ax1.set_xlim([i/ser.sum()*100 for i in axe.get_xlim()])
            ax1.set_xlabel('percentage (%)')
            axe.grid(ax_grid0)
            ax1.grid(ax_grid1)
            axe.set_title(f'{col} Countplot')
        elif top+bottom==top or top+bottom==bottom:
            if top+bottom==top:
                order=ser[:top].index
                s=f'Top {top_}'
            else:
                order=ser[-bottom:].index
                s=f'Bottom {bottom_}'
            sns.countplot(df[col],ax=axe,order=order,hue=hue)
            axe.tick_params('x',labelrotation=90)
            axe.set_title(f'{col} Countplot {s}')

            ax1=axe.twinx()
            ax1.set_ylim([i/ser.sum()*100 for i in axe.get_ylim()])
            ax1.set_ylabel('percentage (%)')
            axe.grid(ax_grid0)
            ax1.grid(ax_grid1)
        else:
            for s,order,ax in zip([f'Top {top_}',f'Bottom {bottom_}'],[ser.index[:top],ser.index[-bottom:]],axe):
                sns.countplot(df[col],ax=ax,order=order,hue=hue)
                ax.tick_params('x',labelrotation=90)
                ax.set_title(f'{col} Countplot {s}')
                ax1=ax.twinx()
                ax1.set_ylim([i/ser.sum()*100 for i in ax.get_ylim()])
                ax1.set_ylabel('percentage (%)')
                ax.grid(ax_grid0)
                ax1.grid(ax_grid1)
    plt.tight_layout()
    plt.show()
        
