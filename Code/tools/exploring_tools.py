#!/usr/bin/env/python

"""
Script with some functions useful for analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from functools import reduce


def missing_data_size(df, figsize=(10, 5), fontsize=15, plot=False, plotname='fig1.png'):
    """
    Function to find the size of the missing data per feature in data frame
    :param df: data frame
    :param figsize: tuple with the size of figure
    :param fontsize: font size
    :param plot: Boolean. True to save figures.
    :param plotname: string with plot name
    :return: Series with features and index and values the total number of fields missing
    """
    try:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        missing.sort_values(ascending=False, inplace=True)
        missing.plot.bar(figsize=figsize, fontsize=fontsize)
        if plot:
            plt.savefig(plotname, bbox_inches='tight', transparent=True)
        else:
            plt.show()
        return missing
    except Exception as e:
        print('Exception: ', e)
        print('Not missing data')


def missing_data_percentage(df, features_missing, figsize=(10, 4), fontsize=15, plot=False, plotname='fig1.png'):
    """
    Function to find the percentage of the missing data per feature in data frame
    :param df: data frame
    :param features_missing: Series with features and index and values the total number of fields missing
    :param figsize: tuple with the size
    :param fontsize: integer
    :param plot: Boolean. True to save figures.
    :param plotname: string with file name
    :return: Data frame
    """
    # Total number of data per feature
    total_data_perfeature = df[features_missing.index.tolist()].isnull().count()
    percentage_data_perfeature = features_missing / total_data_perfeature * 100
    missing_data_features = pd.concat([features_missing, percentage_data_perfeature],
                                      axis=1, keys=['Total', 'Percent'])
    missing_data_features.sort_values(['Total', 'Percent'], ascending=[False, True], inplace=True)
    missing_data_features['Percent'].plot.bar(figsize=figsize, fontsize=fontsize)
    if plot:
        plt.savefig(plotname, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    missing_data_features['total not missing'] = total_data_perfeature - missing_data_features['Total']
    return missing_data_features


def percentage_value_in_feature(df, feature):
    """
    Fundtion that returns the percentage of the values found on a specific feature/column
    :param df: data frame
    :param feature: specific feature/column
    :return: pandas series with data
    """
    total_data_feature = df[feature].isnull().count()
    percentage_data_per_feature = df[feature].value_counts() / total_data_feature
    percentage_data_per_feature.to_frame()
    return percentage_data_per_feature


# Stats


def stats_summary(df):
    """
    Function that gives the statistic summary of the the time serie on demand.
    The data frame should be already as pivote table with products in columns
    and dates in rows.
    Note: We must used funct_pivote_tables_sum_pre() on our data frame first
    that is in the module 'Functions_to_explore'
    :param df: data frame as pivote table
    :return: data frame with summary
    """
    df_trans = df.describe().transpose().copy()
    # Rule of thumb: Having in range of -0.08 to 0.08 skewness sand kurtosis in the range
    # of -3.0 to 3.0 to asses if normality has been achieve by our data
    df_skew = df.skew().to_frame(name='skew')
    # The kurtosis gives an idea of the degree of pickiness of the distribution
    # Positive kurtosis it has more in the tails than the normal distribution.
    # Negative kurtosis, it has less in the tails than the normal.
    df_kurt = df.kurt().to_frame(name='kurt')
    df_median = df.median().to_frame(name='median')
    df_trans['median'] = df_median
    df_trans['skew'] = df_skew
    df_trans['kurt'] = df_kurt
    return df_trans


def plotting_one_dist_norm_kde(df, column, bins=30, fontsize_leg='xx-large', x_label=''):
    """
    Plotting function of distributions and fits.

    :param df: data frame pivoted with the time series and demand on products
    :param column: string. column name
    :param fontsize_leg : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    :param bins: integer. Number of bins.
    :param x_label: string with name of label.
    :return: Arrays containing the x and y values for the fitted KDE line
    """

    if isinstance(df, pd.DataFrame):
        df = df[column]
    mu_s, sigma_s = stats.norm.fit(df)
    desc = df.describe()
    mu = desc.loc['mean']
    sigma = desc.loc['std']
    fig_dist_fit, ax = plt.subplots(figsize=(10, 7))
    # Arrays containing the x and y values for the fitted KDE line
    # The method get lines can be use at once, otherwise the second will overdide the first data
    x_k, y_k = sns.distplot(df, kde=True, bins=bins, ax=ax,
                            hist_kws={'alpha': 0.45, 'color': 'b',
                                      'label': 'Histo: : $\mu=${0:.2f}, $\sigma=${1:.2f})'.format(mu, sigma)},
                            kde_kws={"color": "b", "lw": 1.5, "label": "KDE of {}".format(column)}).get_lines()[
        0].get_data()
    sns.distplot(df, kde=False, ax=ax, fit=stats.norm, color='r', bins=bins,
                 hist_kws={'alpha': 0.0},
                 fit_kws={'color': 'r', 'label': 'Normal: $\mu=${0:.2f}, $\sigma=${1:.2f})'.format(mu_s, sigma_s),
                          'alpha': 0.75}).get_lines()[0].get_data()
    ax.legend(loc='best', fontsize=fontsize_leg)
    ax.tick_params(labelsize=15)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel('PDF', fontsize=15)
    return x_k, y_k


def plotting_one_kde(df, column, ax=None, weights=None, bins=30, fontsize_leg=20,
                     x_label='', loc='best', color='b', case=None, alpha=0.45,
                     bbox_to_anchor=(0, 0, 1, 1), kde=True, hist=True, set_title=False,
                     title='', lw=1.5, weight_y_perce=1, legend_flag=True):
    """
    Function to plot the distribution and the kde.

    :param df: data frame with data
    :param column: string with column name
    :param ax: matplotlib axis, optional if provided, plot on this axis
    :param weights: array_like or None, optional. An array of weights, of the same shape as x.
    Each value in x only contributes its associated weight towards the bin count (instead of 1).
     If normed or density is True, the weights are normalized, so that the integral of the density over
     the range remains 1.Default is None. It can be just applied in histograms not in kde.
    :param bins: argument for matplotlib hist(), or None, optional. Specification of hist bins, or
    None to use Freedman-Diaconis rule.
    :param fontsize_leg: integer
    :param x_label: string with x axis label
    :param loc: :param loc: str or pair of floats, default: rcParams["legend.loc"]
    ('best' for axes, 'upper right' for figures). The strings 'upper left', 'upper right', 'lower left',
    'lower right' place the legend at the corresponding corner of the axes/figure. The strings 'upper center',
    'lower center', 'center left', 'center right' place the legend at the center of the corresponding edge
     of the axes/figure.
    :param color: string indicating the colour.
    :param case: string with name of the case to add in the title.
    :param alpha: scalar, optional, default: None. The alpha blending value, between 0 (transparent) and 1 (opaque).
    :param bbox_to_anchor: BboxBase, 2-tuple, or 4-tuple of floats
    :param kde: Boolean. Whether to plot a gaussian kernel density estimate.
    :param hist: Boolean. Whether to plot a (normed) histogram.
    :param set_title: Boolean indicating if we want to add a title.
    :param title: string with title
    :param lw: float to indicate the line width When we have
    :param weight_y_perce: integer 1 or 100. for showing all in percetage or not.
    :param legend_flag: boolean to let or not the legend automatically
    :return:  Arrays containing the x and y values for the fitted KDE line
    """
    if isinstance(df, pd.DataFrame):
        df = df[column]
    desc = df.describe()
    mu = desc.loc['mean']
    sigma = desc.loc['std']
    median = desc.loc['50%']
    if case is None:
        case = column
    hist_kws = {'alpha': alpha, 'color': color,
                'label': 'Histo: $\mu=${0:.1f}, 50%={1:.1f}, $\sigma=${2:.1f}'.format(mu, median, sigma)}
    kde_kws = {"color": color, "lw": lw, "label": "KDE of {}".format(case)}
    if weights is not None:
        hist_kws['weights'] = weights
        # The KDE is fit with scipy, in displot, but for some reason has not be implemented to pass the
        # weights https://github.com/mwaskom/seaborn/issues/1364
        # However, we use scipy to fit the weight as shown in the documentation
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        # https://stackoverflow.com/questions/53823349/how-can-you-create-a-kde-from-histogram-values-only
        sns.distplot(df, kde=False, hist=hist, bins=bins, ax=ax, hist_kws=hist_kws, norm_hist=True)
        # This part is for getting the x_k for plotting
        h, e = np.histogram(df, bins=bins, density=True)
        x_k = np.linspace(e.min(), e.max())
        # fitting the kde
        kde = stats.gaussian_kde(df,weights=weights)
        y_k = kde.pdf(x_k)
        ax.plot(x_k, y_k * weight_y_perce, '-', color=color, lw=lw, label='KDE of {}'.format(case))
        setting_leg_title_xylabels(ax, bbox_to_anchor, fontsize_leg, loc, set_title, title, x_label,
                                   legend_flag=legend_flag)
    else:
        # Arrays containing the x and y values for the fitted KDE line
        # The method get lines can be use at once, otherwise the second will override the first data

        x_k, y_k = sns.distplot(df, kde=kde, hist=hist, bins=bins, ax=ax,
                                hist_kws=hist_kws,
                                kde_kws=kde_kws).get_lines()[
            0].get_data()
        setting_leg_title_xylabels(ax, bbox_to_anchor, fontsize_leg, loc, set_title, title, x_label,
                                   legend_flag=legend_flag)
    return x_k, y_k


def setting_leg_title_xylabels(ax, bbox_to_anchor, fontsize_leg, loc, set_title, title, x_label, legend_flag=True):
    """
    Function to set the legend, title, and xy labels for plotting_one_kde.

    :param ax:  matplotlib axis, optional if provided, plot on this axis
    :param bbox_to_anchor: BboxBase, 2-tuple, or 4-tuple of floats
    :param fontsize_leg: integer:
    :param loc: :param loc: str or pair of floats, default: rcParams["legend.loc"]
    ('best' for axes, 'upper right' for figures). The strings 'upper left', 'upper right', 'lower left',
    'lower right' place the legend at the corresponding corner of the axes/figure. The strings 'upper center',
    'lower center', 'center left', 'center right' place the legend at the center of the corresponding edge
     of the axes/figure.
    :param set_title: Boolean indicating if we want to add a title.
    :param title: string with title
    :param x_label: string with x axis label
    :param legend_flag: boolean to let or not the legend automatically
    """
    if legend_flag:
        ax.legend(loc=loc, fontsize=fontsize_leg * 0.6, bbox_to_anchor=bbox_to_anchor)
    ax.tick_params(labelsize=fontsize_leg * 0.8)
    ax.set_xlabel(x_label, fontsize=fontsize_leg * 0.8)
    ax.set_ylabel('PDF [%]', fontsize=fontsize_leg * 0.8)
    if set_title:
        ax.set_title(title, fontsize=fontsize_leg)


def multiple_plot_distr(df, figsize=(6, 6), ncols=2, nrows=2):
    """
    Function that makes subplots of all the distributions and fits
    using the plotting_one_dist_norm_kde function for a single plot.
    It depends on function 'plotting_one_dist_norm_kde'
    :param df: data frame pivoted with the time series and demand on products
    :param figsize: tuple with the size
    :param ncols: number of columns in the subplot
    :param nrows: number of rows in the subplot
    """
    fig, axs = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows)
    columns_list = df.columns.tolist()
    # len_items_col = len(columns_list)
    for row in range(nrows):
        for col in range(ncols):
            plotting_one_dist_norm_kde(df, columns_list[col], bins=30)
        # Removes from the list the ones that have been plotted
        columns_list.pop(row)


def q_q_plotting(df, column, dist="norm", plot=plt, fit=True):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html
    """
    Calculate quantiles for a probability plot, and optionally show the plot.
    :param df: data frame
    :param column: string, column name
    :param dist:  str or stats.distributions instance, optional
    Distribution or distribution function name. The default is ‘norm’ for a normal probability plot.
    Objects that look enough like a stats.distributions instance (i.e. they have a ppf method) are also accepted.
    :param plot: object, optional
    If given, plots the quantiles and least squares fit. plot is an object that has to have methods “plot” and “text”.
    The matplotlib.pyplot module or a Matplotlib Axes object can be used, or a custom object with the same methods.
    Default is None, which means that no plot is created.
    :param fit : bool, optional
    Fit a least-squares regression (best-fit) line to the sample data if True (default).
    :return: (osm, osr) : tuple of ndarrays
    Tuple of theoretical quantiles (osm, or order statistic medians) and ordered responses (osr).
    osr is simply sorted input x. For details on how osm is calculated see the Notes section.
    (slope, intercept, r) : tuple of floats, optional Tuple containing the result of the least-squares fit,
    if that is performed by probplot. r is the square root of the coefficient of determination.
    If fit=False and plot=None, this tuple is not returned.
    """
    if isinstance(df, pd.DataFrame):
        df = df[column]
    return stats.probplot(df, dist=dist, plot=plot, fit=fit)


def fit_normal_distribution(df, column):
    """
    Fit the parameters of a normal distribution based on the data and plot it
    :param df: data frame pivoted with the time serie and demand on products
    :param column: column of the line_number/ EU number of the product
    :return: array and data frame of values of the fitted data
    """
    mean, std = stats.norm.fit(df[column])
    plt.hist(df[column], bins=30, normed=True)
    xmin, xmax = plt.xlim()
    plt.xlim(xmin - 5, xmax + 5)
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    df_norm_dist = pd.DataFrame({column: y})
    plt.plot(x, y)
    plt.title(' Mean:{:2f}; sdt:{:2f}'.format(mean, std))
    return y, df_norm_dist


def convert_to_standard_normal(df, column):
    """
     Function to normalise data
    :param df: data frame pivoted with the time serie and demand on products
    :param column: column of the line_number/ EU number of the product
    :return: data frame with the data transformed to standard normal
    """
    df_norm = (df[column] - np.mean(df[column])) / np.std(df[column])
    return df_norm.to_frame()


# Plotting


def bar_plot_quick(series, figsize=(10, 5), fontsize=15, vert=True):
    """
    Quick barplots
    :param series: pandas series
    :param figsize: tuple with size (x,y)
    :param fontsize: integer
    :param vert: boolean. True the bars are vertical
    """
    if vert:
        try:
            series.plot.bar(figsize=figsize, fontsize=fontsize)
            return series
        except Exception as e:
            print('There is  problem: ', e)
    else:
        try:
            series.plot.barh(figsize=figsize, fontsize=fontsize)
            return series
        except Exception as e:
            print('There is  problem: ', e)


def ecdf(series, grid=False, x_label="# of cases"):
    """
    Function to get the x, y for the Empirical CDF
    The CDF gives the probability that the measured quantity
    will be less than the value on the x-axis
    :param series: pandas series
    :param grid: Boolean to indicate to add grid
    :param x_label: string with name of label
    :return: numpy arrays of sorted data
    """
    # number of data points
    n = len(series)
    # x-data  for ECDF
    x = np.sort(series)
    # y-data for ECDF
    y = np.arange(1, n + 1) / n
    plt.plot(x, y, marker='.', linestyle='none')
    plt.xlabel(x_label)
    plt.ylabel('ECDF')
    plt.grid(grid)
    return x, y


def ecdf_ax(series, ax, grid=False, fontsize=20,
            x_label="# of cases"):
    """
    Function to get the x, y for the Empirical CDF
    The CDF gives the probability that the measured quantity
    will be less than the value on the x-axis
    :param series: pandas series
    :param ax:  matplotlib axis, optional if provided, plot on this axis
    :param grid: Boolean to indicate to add grid
    :param x_label: string with name of label
    :param fontsize: int indicating font size
    :return: numpy arrays of sorted data
    """
    # number of data points
    n = len(series)
    # x-data  for ECDF
    x = np.sort(series)
    # y-data for ECDF
    # This cumulative distribution function is a step function
    # that jumps up by 1/n at each of the n data points
    y = np.arange(1, n + 1) / n
    ax.plot(x, y, marker='.', linestyle='none')
    ax.set_xlabel(x_label, fontsize=fontsize * 0.8)
    ax.set_ylabel('ECDF', fontsize=fontsize * 0.8)
    ax.grid(grid)
    return x, y


def ecdf_hist(series, bins=40, figsize=(10, 5), grid=True, x_label="# of cases"):
    """
    Function to get the histogram x, y for the Empirical CDF
    The CDF gives the probability that the measured quantity
    will be less than the value on the x-axis
    :param series: pandas series
    :param bins: integer with number of bins
    :param figsize: tuple with the size
    :param grid: Boolean to indicate to add grid
    :param x_label: string with name of label
    """
    series.hist(bins=bins, color='b', figsize=figsize, cumulative=True, density=True)
    plt.grid(grid)
    plt.xlabel(x_label)
    plt.ylabel('ECDF')


def quick_hist(series, fontsize=18, bins=20, title='', x_label='', y_label='',
               plot_stats=False, x_stats=0.8, y_stats=0.8, grid=False, figsize=(10, 5), **kwargs):
    """
    Function to plot a quick histogram or cumulative (ECDF).

    :param series: pandas series
    :param fontsize: int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    :param bins: integer with number of bins
    :param title: string with title
    :param x_label:string with name of label
    :param y_label:string with name of label
    :param plot_stats: Boolean to indicate to calculate and plot the mean and median.
    :param x_stats: float. This value helps to move in the x axis the Text for the mean value on the plot.
    :param y_stats: float. This value helps to move in the y axis the Text for the mean value on the plot.
    :param grid: Boolean to indicate to add grid
    :param figsize: tuple with the size
    :param kwargs: Properties. Allow you to pass a variable number of arguments to a function.
    """
    series.hist(bins=bins, figsize=figsize, **kwargs)
    if plot_stats:
        min_ylim, max_ylim = plt.ylim()
        mean = series.mean()
        median = series.median()
        plt.axvline(mean, color='g', linestyle='dashed', linewidth=2)
        plt.axvline(median, color='orange', linestyle='dashed', linewidth=2)
        plt.text(mean * x_stats, max_ylim * y_stats, 'Mean: {:.2f}'.format(mean), color='g')
        plt.text(mean * x_stats, max_ylim * (y_stats - 0.05), 'Median: {:.2f}'.format(median), color='orange')
    plt.xlabel(x_label, fontsize=fontsize * 0.85)
    plt.ylabel(y_label, fontsize=fontsize * 0.85)
    plt.xticks(fontsize=fontsize * 0.85)
    plt.yticks(fontsize=fontsize * 0.85)
    plt.title(title, fontsize=fontsize, pad=15)
    plt.grid(grid)


def to_datetime_trans(df, list_columns, infer_datetime_format=False, format_date='%d/%m/%Y'):
    """
    Transforming to datetime some columns
    :param df: data frame
    :param list_columns:  list of features to transform.
    :param infer_datetime_format: Boolean. True when inferring datetime.
    :param format_date: string with datetime format
    """
    for feature in list_columns:
        if infer_datetime_format:
            df[feature] = pd.to_datetime(df[feature], infer_datetime_format=infer_datetime_format)
        else:
            df[feature] = pd.to_datetime(df[feature], format=format_date)


def feature_compare(df_1, df_2, feat_list, float_comp=False, atol=1e-08):
    """
    Method to compare values of two columns in different data frames with the same index. They can be strings, integers, floats. For the latter,
    we need to specify that we are comparing floats and we provide the tolerance value for the
    comparison.

    param df_1: data frame.
    param df_2: data frame.
    param feat_list[0]: string of names of columns. they can have the same name or different.
    param float_comp: Boolean to activate float comparison
    param atol:  float, The absolute tolerance parameter.
    return data frame with the cases that were not macthing.
    """
    df_c1 = df_1.copy()
    df_c2 = df_2.copy()
    if float_comp:
        value_compare = np.isclose(df_c1[feat_list[0]], df_c2[feat_list[1]], atol=atol)
        df_c1['comp'] = value_compare
        perc1 = df_c1['comp'].value_counts() / df_c1.shape[0] * 100
        print(perc1)
        false_matching = df_c1[df_c1['comp'] == False]
        del false_matching['comp']
    else:
        value_compare = df_c1[feat_list[0]] == df_c2[feat_list[1]]
        perc = value_compare.value_counts() / df_c1.shape[0] * 100
        print(perc)
        false_matching = df_c1[~value_compare]
    return false_matching


def merge_multiple(list_dfs, on, **kwargs):
    """
    Function to merge many data frames
    :param list_dfs: list with pandas data frames to merge
    :param on: string, column name for mergin on
    :param kwargs: merge parameters
    :return: padnas data frame
    """
    return reduce(lambda left, right: pd.merge(left, right, on=on, **kwargs), list_dfs)


def heat_map_corr(df, cmap='coolwarm', figsize=(20, 15), annot=False, font_annot=8, fontsize=12, square=True):
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.heatmap(df,
                    cmap=cmap,
                    ax=ax,
                    annot=annot,
                    fmt=".1%",
                    linewidths=0.5,
                    square=square,
                    annot_kws={"size":font_annot})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=fontsize)
    g.set_yticklabels(g.get_xmajorticklabels(), fontsize=fontsize)


def corr_func(x, y, person=True):
    """
    Function to calculate the correlation and p-value to test for non-correlation.
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
    :param x: array
    :param y: array
    :param person: boolean. When false we use the spearman coef
    :param kws: optional keywords (we need it when calling it with facet grid)
    :return: correlation coefficient and the p-value to test for non-correlation
    """
    if person:
        r, p = stats.pearsonr(x, y)
        return r, p
    else:
        r, p = stats.spearmanr(x, y)
        return r, p


def corr_func_plot(x, y, person=True, p_val=False):
    """
    Function to add the pearson/spearman coefficients in the Facet grid
    :param x: array
    :param y: array
    :param person: boolean. When false we use the spearman coef
    :param p_val: Boolean. True to print p value in plot.
    :param kws: optional keywords (we need it when calling it with facet grid)
    """
    r, p = corr_func(x, y, person=person)
    ax = plt.gca()
    if p_val:
        ax.annotate("p = {:.3f}".format(p), xy=(.4, .9), xycoords=ax.transAxes)
        ax.annotate("r = {:.2f}".format(r), xy=(.1, .9), xycoords=ax.transAxes)
    else:
        ax.annotate("r = {:.2f}".format(r), xy=(.1, .9), xycoords=ax.transAxes)


def facet_grid_dist(data,
                    lower=sns.regplot,
                    upper=plt.scatter,
                    diag=plt.hist,
                    p_val=False,
                    person=True, bins=20):
    """
     FacetGRid to check distribution and relationship among features

    :param data: pandas data frame
    :param lower: sns o plt. plot type object e.e.sns.kdeplot
    :param upper: sns o plt. plot type object
    :param diag: sns o plt. plot type object
    :param p_val: Boolean. True to print p value in plot.
    :param person: Boolean. False to use Spearman coefficient
    :param bins: int. Indicate number of bins for the distribution
    """
    # Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
    figs_corr = sns.PairGrid(data)
    # Using map_upper we can specify what the upper triangle will look like.
    figs_corr = figs_corr.map_upper(upper, color='purple')
    # We can also define the lower triangle in the figure, inclufing the plot type (kde) or the color map (BluePurple)
    figs_corr = figs_corr.map_lower(lower)
    # figs_corr =figs_corr.map_lower(sns.kdeplot, cmap='cool_d')
    # Finally we'll define the diagonal as a series of histogram plots of the daily return
    figs_corr = figs_corr.map_diag(diag, bins=bins)
    # plotting correlation value and p value
    figs_corr = figs_corr.map(corr_func_plot, p_val=p_val, person=person)