import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def verification(df: pd.DataFrame, path: str):
    """
        Check if the dataframe has more than two columns, without empty labels in the 1st column,
    or columns with less than 3 valid data.
        Check if labels are all identical and remove labels with less than 5 samples.
    :param df: pandas.Dataframe.
    :param path: Output directory.
    :return: verified dataframe, headers (list)
    """
    print(">>>>> Data verifying >>>>>")

    if df.shape[1] < 2:  # check if less than 2 columns
        raise ValueError("At least two columns should be included")

    _features = df.columns  # features
    _col_cls = df.iloc[:, 0]  # labels
    _cls_counts = _col_cls.value_counts()
    _cls_num = _col_cls.nunique()

    # check labels
    if _col_cls.isna().any():  # check if empty label exists
        raise ValueError("Empty labels in the 1st column")
    if _cls_num < 2:  # check if all labels are the same
        raise ValueError("Labels should have more than one class")

    # remove class with less than 5 duplicates
    _idx_drop = _cls_counts[_cls_counts < 5].index  # find
    if not _idx_drop.empty:
        df_dup5 = df[~df.iloc[:, 0].isin(_idx_drop)]  # remove
        df = df_dup5.reset_index(drop=True)  # update index
        print(f"The following classes are dropped due to less than 5 duplicates:\n{_idx_drop.values}\n")

    # remove columns with less than 3 values
    _col_keep = df.notnull().sum()
    _col_drop = _col_keep[_col_keep < 3].index
    if not _col_drop.empty:
        df = df.drop(columns=_col_drop)
        print(f"The following columns are dropped due to less than 3 valid data:\n{_col_drop.values}\n")

    excel_out = os.path.join(path, "Results.xlsx")
    df_cls_info = pd.DataFrame({'Class': _cls_counts.index, 'Counts': _cls_counts.values})
    with pd.ExcelWriter(excel_out, engine='openpyxl', mode='w') as file_excel:
        df_cls_info.to_excel(file_excel, index=False, sheet_name='Class_info')

    print("<<<<< Data verified  <<<<<")
    return df, _features


def normalization(
        x: pd.DataFrame,
        y: pd.DataFrame,
        path: str = None,
        x_target: pd.DataFrame = None,
        y_target: pd.DataFrame = None,
        corr_threshold: float = 0.9,
        cat_encoder: str = 'OrdinalEncoder',
):
    """
    Impute missing data, encodes the categorical data, and scale the numerical data to [0,1].
    Output the normalized data and their Pearson correlation coefficient.
    :param x: Basic features to be fitted.
    :param y: Labels of *x*.
    :param path: Output directory.
    :param x_target: Target features to be transformed.
    :param y_target: Labels of *x_target*. If *x_target* is not None, this has to be provided.
    :param corr_threshold: Threshold of correlation coefficient for highly correlated features.
    :param cat_encoder: Estimator used to encode categorical data. 'OrdinalEncoder' or 'OneHotEncoder'
    :return: Normalized dataframe.
    """
    if x_target is None:
        x_target = x
        _display = 'Training Data'
        _sheets = ['Normalized', 'PCC']
    else:
        if y_target is None:
            raise ValueError("Labels of features to be transformed are needed")
        _display = 'Target Data'
        _sheets = ['Normalized_tar', 'PCC_tar']

    # Check empty labels
    if not len(x) == len(y):
        raise ValueError("Features for some labels are empty")

    print(f">>>>> {_display} normalizing >>>>>")

    # numerical columns
    _df_num = x.select_dtypes(include=['number'])
    col_num = _df_num.columns.values
    # categorical columns
    _df_cat = x.select_dtypes(include=['object'])
    col_cat = _df_cat.columns.values

    # display
    if col_num.size == 0:
        print("***** No numerical data *****")
    if col_cat.size == 0:
        print("***** No categorical data *****")

    # check if there are numerical values in categorical columns
    for i_col in col_cat:
        numeric_rows = pd.to_numeric(_df_cat[i_col], errors='coerce').notnull()  # 无法转换为数值类型的数据转替换为NaN
        if numeric_rows.any():
            raise ValueError("Numerical values in categorical columns:\n"
                             f"{_df_cat[i_col][numeric_rows]}\n")

    # imputer, encoder and scaler
    if cat_encoder == "OneHotEncoder":
        from sklearn.preprocessing import OneHotEncoder
        transform_cat = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ])
    elif cat_encoder == "OrdinalEncoder":
        from sklearn.preprocessing import OrdinalEncoder
        transform_cat = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
            ('encoder', OrdinalEncoder(
                categories='auto',
                dtype=int,
                handle_unknown='use_encoded_value',
                unknown_value=len(y),
            )),  # 整数编码
            ('scaler', MinMaxScaler())
        ])
    else:
        raise ValueError("Unsupported encoder")

    # normalize
    transform_num = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', transform_num, col_num),
        ('cat', transform_cat, col_cat)
    ])
    preprocessor.fit(x)
    data_scaled = preprocessor.transform(x_target)

    # remove invariant columns
    sel = VarianceThreshold(threshold=0)
    sel.fit_transform(data_scaled)

    # generate dataframe
    if cat_encoder == "OneHotEncoder":
        df_scaled = pd.DataFrame(data_scaled)

        for i in range(len(col_num)):  # rename columns
            df_scaled.rename(columns={df_scaled.columns[i]: col_num[i]}, inplace=True)
    elif cat_encoder == "OrdinalEncoder":
        df_scaled = pd.DataFrame(data_scaled, columns=np.concatenate([col_num, col_cat]))
    else:
        raise ValueError("Unsupported encoder")

    # Correlation efficient
    styled_corr, df_corr = correlation(df_scaled, corr_threshold)

    if path is not None:  # output results if offering *path*
        # write to Excel
        excel_out = os.path.join(path, "Results.xlsx")
        with pd.ExcelWriter(excel_out, engine='openpyxl', mode='a', if_sheet_exists='replace') as file_excel:
            if y_target is None:
                df_out = pd.concat([y, df_scaled], axis=1)  # convert to dataframe
            else:
                df_out = pd.concat([y_target, df_scaled], axis=1)  # convert to dataframe

            df_out.to_excel(file_excel, index=False, sheet_name=_sheets[0])  # output
            styled_corr.to_excel(file_excel, index=True, sheet_name=_sheets[1])

    # Check highly correlated features
    if y_target is None:
        high_corr = ((abs(df_corr) > corr_threshold) & (abs(df_corr) < 1)).any().any()
        if high_corr:
            raise ValueError("Highly correlated features in training data. "
                             "Please check worksheet 'PCC' in 'Result.xlsx' and keep only one of those")

    print(f"<<<<< {_display} normalized  <<<<<")
    return df_scaled


def correlation(df: pd.DataFrame, threshold: float):
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold should between [0, 1]")

    df_corr = df.corr(
        method='pearson',  # 相关性分析算法
        numeric_only=False  # 只分析数值型数据
    )
    corr_styled = df_corr.style.map(lambda cell: 'background-color: red' if abs(cell) > threshold else '')

    return corr_styled, df_corr
