#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import joblib as jl
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta as rd
import seaborn as sns
from matplotlib.widgets import RadioButtons
from statsmodels.robust.robust_linear_model import RLM
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib

# set up matplotlib
gui_env = ['Qt5Agg', 'TKAgg', 'Qt4Agg', 'GTKAgg', 'WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui, warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print("Using:", matplotlib.get_backend())

# set up seaborn and other variables
sns.set()


# modify the statsmodels mode function
def mode(x): return stats.mode(x)[0]


# setup number of clusters and random state
n_clusters = 5
prng_seed = 1

# read in the data
basedf = pd.read_csv('online_retail.csv', parse_dates=['InvoiceDate'])
print(basedf.shape)

df = basedf
df["Total_Price"] = df["Quantity"] * df['Price']

# reduce the accuracy of dates to day
df['InvoiceDate'] = df.InvoiceDate.values.astype('datetime64[D]')

# create a reference point for calculating recency and first_purchase
NOW = df.InvoiceDate.max() + rd(months=1)

print(df.head(10))

print(df.Country.unique())
print(df.Country.nunique())

# let's focus on the country with biggest customer base
df = df.loc[df['Country'] == 'United Kingdom']

# remove NA's and nulls
df.isnull().sum(axis=0)
df = df[pd.notnull(df['Customer ID'])]
print(len(np.unique(df['Customer ID'])))

# let's remove some outliers
df = df[(df['Quantity'] > 0) & (df['Price'] > .001)]
print(df.describe())

# let's plot Quantity and Price
plt.figure()
sns.scatterplot(df.Quantity, df.Price)
plt.tight_layout()
plt.show()

# remove some more outliers
df = df[(df['Quantity'] < 6000) & (df['Price'] < 1500) & (df['Customer ID'] != 18102)]
print(df.shape)
df = df[~df.StockCode.str.contains('TEST')]
print(df.describe())

# unpack the data using log function and run a robust linear regression model
x = np.log(df.Quantity)
y = np.log(df.Price)
results = RLM(y, sm.add_constant(x.values)).fit()
print(results.summary2())

# plot the data with the regression line
plt.figure()
sns.lineplot(x, (results.params[1] * x + results.params[0]))
sns.scatterplot(x, y)
plt.tight_layout()
plt.show()

# df[df['Customer ID'] == 12748].InvoiceDate.value_counts()

# summarise InvoiceDate and Total Price by Customer ID and Invoice Date
# multiple invoices in a day will still count to 1
# and rename the columns
rfmMonthly = df.groupby(['Customer ID', 'InvoiceDate']).agg({
    'InvoiceDate': lambda x: len(np.unique(x)),
    'Total_Price': 'sum',
}).rename(columns={'InvoiceDate': 'Frequency', 'Total_Price': 'Monetary'}).reset_index()

# oversample the data to monthly time period
# filter out extra dates created by pandas
rfmMonthly = rfmMonthly.groupby(['Customer ID']).resample('M', on='InvoiceDate').sum()
rfmMonthly.drop(columns='Customer ID', inplace=True)
rfmMonthly = rfmMonthly[rfmMonthly['Frequency'] != 0].reset_index()

print(rfmMonthly.head(10))

# Finally calculate RFM nad First_Purchase metrics for the monthly data
rfmdf = pd.DataFrame()
gb = rfmMonthly.groupby('Customer ID')
rfmdf['Recency'] = gb['InvoiceDate'].agg(lambda x: np.round((NOW - x.max()).days / 30, 2))
rfmdf['Frequency'] = gb['Frequency'].agg('sum')
rfmdf['Monetary'] = gb['Monetary'].agg('sum')
rfmdf['First_Purchase'] = gb['InvoiceDate'].agg(lambda x: np.round((NOW - x.min()).days / 30, 2))

print(rfmdf.head(10))

# normalize the data for KMeans
scaler = StandardScaler()
rfmf = scaler.fit_transform(rfmdf)
kcluster = KMeans(n_clusters=n_clusters, init='random', n_jobs=-1, random_state=prng_seed)

# run the KMeans and pickle the data using joblib
# klist file stores the cluster allocations
# and kinertia file stores the MSE aka inertia
ignore_files = False
klistfile = Path('klist.pickle')
kinertiafile = Path('kinertia.pickle')
if klistfile.exists() and kinertiafile.exists() and not ignore_files:
    klist = jl.load(klistfile)
    kinertia = jl.load(kinertiafile)
else:
    klist = list()
    kinertia = list()
    for i in range(1000):
        rfm = shuffle(rfmf, random_state=i)
        klist.append(kcluster.fit_predict(rfm))
        kinertia.append(kcluster.inertia_)
    jl.dump(klist, klistfile, compress=True)
    jl.dump(kinertia, kinertiafile, compress=True)
karr = np.array([np.unique(x, return_counts=True)[1] for x in klist])

print(kinertia.index(min(kinertia)))
print(min(kinertia))

print("--------Variation in number of customer ids assigned to a cluster---------------")
print("Mean: ", np.round(np.mean(karr, axis=0), 2))
print(" Std: ", np.round(np.std(karr, axis=0, ddof=1), 2))

# assign the clusters
# add 1 so that the cluster sequence starts from 1
rfmfdf = shuffle(rfmdf, random_state=kinertia.index(min(kinertia)))
rfmfdf['clusters'] = klist[kinertia.index(min(kinertia))] + 1

# calculate total sales for contribution from each cluster
Total_sales = np.sum(rfmfdf.Monetary.values)
print(Total_sales)

# calculate percentage contribution from each cluster
clust_sales = rfmfdf.groupby('clusters').agg(
    {'Monetary': lambda x: x.sum() / Total_sales * 100}).reset_index()
clust_sales.rename(columns={'Monetary': 'clust_sales_contrib'}, inplace=True)
print(clust_sales)

# merge clust_sales with rfmdf to assign each cluster its sales contrib
rfmfdf.reset_index(inplace=True)
rfmTable = rfmfdf.merge(clust_sales, how='inner', on='clusters').set_index('Customer ID')
rfmTable.head()

# create some plots to help in final analysis
# %matplotlib inline
fig = plt.figure()
ax = fig.add_subplot(111)
cols = ['Recency', 'Frequency', 'Monetary', 'First_Purchase']
i = 0
# sns.distplot(rfmTable[cols[i]],kde = False,color='b')
n = plt.hist(rfmTable[cols[i]], bins=5)
ax.set_xlabel(cols[i])
plt.tight_layout()
plt.show()

fig = plt.figure()
cols = ['Recency', 'Frequency', 'Monetary', 'First_Purchase']
for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1)
    sns.distplot(rfmTable[cols[i]], kde=False)
    #     plt.hist(rfmTable[cols[i]])
    ax.set_xlabel(cols[i])
plt.tight_layout()
plt.show()

rfm_stats = np.round(rfmTable.melt(id_vars=['clusters'], var_name='Metrics').groupby(['clusters', 'Metrics']).agg(
    ['min', 'mean', mode, 'max']), 2)
rfm_stats.columns = ['Minimum', 'Mean', 'Mode', 'Maximum']
rfm_stats = rfm_stats.reindex(axis='index', level=1, labels=['Recency', 'Frequency', 'Monetary',
                                                             'First_Purchase', 'clust_sales_contrib'])
print(rfm_stats)

cluster = rfmTable
plt.figure()
ax = plt.axes(projection='3d')
zdata = cluster.Monetary
ydata = cluster.Frequency
xdata = cluster.Recency
ax.scatter3D(xdata, ydata, zdata, c=zdata, s=50)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(16, 9))
# fig = plt.figure()
for i in range(n_clusters + 1):
    ax_ = fig.add_subplot(3, 2, i + 1, projection='3d')
    if not (i):
        cluster = rfmTable
        title = "All"
    else:
        cluster = rfmTable[rfmTable.clusters == i]
        title = "Cluster " + str(i)
    zdata = cluster.Monetary
    ydata = cluster.Frequency
    xdata = cluster.Recency
    ax_.scatter3D(xdata, ydata, zdata, c=zdata, s=50)
    ax_.set_xlabel('Recency')
    ax_.set_ylabel('Frequency')
    ax_.set_zlabel('Monetary')
    ax_.set_title(title)
plt.tight_layout()
plt.show()


def histogramfn(metric):
    hist, xedges, yedges = np.histogram2d(rfmTable[metric], rfmTable.clusters, bins=5)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    # Construct arrays with the dimensions for the bars.
    dx = 1.5 * np.ones_like(zpos)
    dy = 0.9 * np.ones_like(zpos)
    dz = hist.ravel()
    return xpos, ypos, zpos, dx, dy, dz


metrics = ['Recency', 'Frequency', 'Monetary', 'First_Purchase']
metric = metrics[0]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
rax = plt.axes([0.15, 0.7, 0.15, 0.15])
radio = RadioButtons(rax, metrics)
xpos, ypos, zpos, dx, dy, dz = histogramfn('Recency')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
ax.set_xlabel(metric)
ax.set_ylabel('Clusters')
ax.set_yticks(range(1, n_clusters + 1), minor=False)


def bar_3d(metric):
    ax.clear()
    xpos, ypos, zpos, dx, dy, dz = histogramfn(metric)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    ax.set_xlabel(metric)
    ax.set_ylabel('Clusters')
    ax.set_yticks(range(1, n_clusters + 1), minor=False)
    plt.draw()


radio.on_clicked(bar_3d)
plt.grid()
plt.show()
