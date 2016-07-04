# kaggle

## 1. Data Analysis 
> import pandas as pd

[pandas.DataFrame.drop](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html): Return new object with labels in requested axis removed.

[pandas.DataFrame.sort_values](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html?highlight=sort_values#pandas.DataFrame.sort_values):Sort by the values along either axis.

[pandas.isnull](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.isnull.html):Detect missing values (NaN in numeric arrays, None/NaN in object arrays).

[pandas.Series.value_counts](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html?highlight=value_counts):Returns object containing counts of unique values.

[pandas.DataFrame.apply](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html?highlight=apply#pandas.DataFrame.apply):Applies function along input axis of DataFrame.

> Objects passed to functions are Series objects having index either the DataFrame’s index (axis=0) or the columns (axis=1). Return type depends on whether passed function aggregates, or the reduce argument if the DataFrame is empty.

[pandas.get_dummies](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies):Convert categorical variable into dummy/indicator variables.

[pandas.concat](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html?highlight=concat#pandas.concat):Concatenate pandas objects along a particular axis with optional set logic along the other axes. Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.

