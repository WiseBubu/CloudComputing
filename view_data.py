#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os
import dash
# import dash_html_components as html
# import dash_core_components as dcc
from dash import dcc
from dash import html
from dash.dependencies import  Input, Output
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import scipy.stats as st
from statsmodels.graphics.gofplots import qqplot
import plotly.figure_factory as ff
import boto3
#%%
#path = 'D:\GWU\Complex Data Visualization\Dataset\\archive\\taxi_final.csv'
s3_client = boto3.client('s3')
s3_bucket_name = 'projectdataforcc2022'
s3 = boto3.resource('s3',
                    aws_access_key_id= 'AKIA6HGIBOFCOKFBLSKY',
                    aws_secret_access_key='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')

my_bucket=s3.Bucket(s3_bucket_name)
# bucket_list = []
# for file in my_bucket.objects.all():
#     file_name=file.key
#     if file_name.find(".csv")!=-1:
#         bucket_list.append(file.key)
# length_bucket_list=print(len(bucket_list))
#%%
# import sys
# if sys.version_info[0] < 3:
#     from StringIO import StringIO  # Python 2.x
# else:
#     from io import StringIO  # Python 3.x
# import io
# df = []  # Initializing empty list of dataframes
# for file in bucket_list:
#     obj = s3.Object(s3_bucket_name, file)
#     data = obj.get()['Body'].read()
#     df = pd.read_csv(io.BytesIO(data))#, header=0, delimiter=",")
   # df.append(pd.read_csv(io.BytesIO(data), header=0, delimiter=",", low_memory=False))
obj = s3_client.get_object(
    Bucket=s3_bucket_name,
    Key='taxi_data.csv'
)

data = pd.read_csv(obj['Body'])
# ll = []
# for i, chunks in enumerate(pd.read_csv(obj['Body'],chunksize=100)):
#     if i >2:
#         break
#     ll.append(chunks)
#%%
# path = os.getcwd()+'/taxi_final.csv'
#ant_list=['PROVIDER NAME','StartDateTime','FareAmount','GratuityAmount','SurchargeAmount','ExtraFareAmount','TollAmount','TotalAmount','PaymentType','EndDateTime','OriginState','DestinationState','Milage','Duration']
want_list=['PROVIDER NAME','FareAmount','GratuityAmount','SurchargeAmount','ExtraFareAmount','TollAmount','TotalAmount','PaymentType','OriginState','DestinationState','Milage','Duration']
# df2 = pd.read_csv(path,usecols=want_list)
#%%
# df2.describe()
#%%
df2 = data.copy()
df2 = df2.dropna()
df2.PaymentType = df2.PaymentType.astype('str')
#%%
count_num=0
count_obj=0
category_list = []
feature_list = []
float_list = []
for i in df2.columns:
    if df2[i].dtype == 'float':
        count_num +=1
        feature_list.append(i)
        float_list.append(i)
    else:
        count_obj += 1
        category_list.append(i)
print(count_obj)
print(count_num)
target = feature_list.pop(-3)
#%%
drop_feature_list=[]
for i in range(len(feature_list)):
    drop_feature_list.append({'label':i,'value':i})
drop_feature_list.append({'label':'mle','value':'mle'})
#%%
# Furthermore, remove non values in these two columns
df2 = df2[(df2['OriginState']!='--') & (df2['DestinationState']!='--')&(df2['DestinationState']!='  ')]
#%%
drop_list_boxplot=[]
for i in want_list:
    if df2[i].dtype == 'float':
        drop_list_boxplot.append({'label':i,'value':i})
#%%
drop_list_all = []
for i in want_list:
    drop_list_all.append({'label':i,'value':i})
# # Pop time from the dict
# drop_list_all.pop(9)
# drop_list_all.pop(1)
#%%
# category_list = ['PROVIDER NAME','PaymentType','OriginState','DestinationState']
category_list.append('None')
drop_list_category=[]
for i in category_list:
    drop_list_category.append({'label': i, 'value': i})

#%%
for i in df2.columns:
    if df2[i].dtype!='float':
        print(f'{df2[i].unique()}')
#%%
df21 = df2.iloc[0:50000,]
df22 = df2.copy()

# Only keep paymentType 1 and 2 for settling imbalance (only a few 3 and 4)
df22 = df22[(df22['PaymentType']=='1.0')|(df22['PaymentType']=='2.0')]
df23 = df22.sample(20000)
#%%
def remove_outliers_iqr(df22,col):
    for i in col:
        print(i)
        q1 = df22[i].quantile(0.25)
        q3 = df22[i].quantile(0.75)
        iqr = q3-q1
        df22 = df22[(df22[i]>=q1-1.5*iqr)&(df22[i]<=q3+1.5*iqr)]
        print(len(df22))
        print('='*100)
    return df22
#%%
# Because this is a large dataset, theres needs to be more times for cleaning outliers.
for i in range(4):
    df22 = remove_outliers_iqr(df22,float_list)

# Remove outliers by looking at boxplots and clean the dataset manually.
# df22 = df2[(df2['FareAmount']<15.94)&(df2['FareAmount']>0.75)&(df2['GratuityAmount']<4)&(df2['SurchargeAmount']<0.3)&(df2['SurchargeAmount']>0)&(df2['ExtraFareAmount']<2)&(df2['ExtraFareAmount']>=0)&(df2['TollAmount']<40)&(df2['TollAmount']>=0)&(df2['TotalAmount']<18.22)&(df2['TotalAmount']>1.75)&(df2['Milage']<3.5)&(df2['Duration']>0)&(df2['Duration']<22)]
#%%
# Statistics
for i in float_list:
    print(f'The mean of {i} is {df22[i].mean():.2f}')
    print(f'The variance of {i} is {np.var(df22[i]):.2f}')
    print(f'The median of {i} is {np.median(df22[i]):.2f}')
    print('='*100)
#%%
# Get scatter matrix
# fig = px.scatter_matrix(df22.sample(20000),
#              dimensions=float_list)
# fig.show(renderer= 'browser')
#%%
# Get qqplot
fig = qqplot(df22.FareAmount,line='s')
plt.title('qqplot of FareAmount')
fig.show()
#%%
# kde plot
sns.displot(data= df22, x = 'FareAmount',hue='PaymentType',stat='density',kde=True)
plt.show()
#%%
# Subplots to show the ratio of provider name
pie_temp = []
list_temp1 = df22['PROVIDER NAME'].unique()
for i in df22['PROVIDER NAME'].unique():
    pie_temp.append(len(df22[df22['PROVIDER NAME']==i]))
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.hist(df22['PROVIDER NAME'])
plt.xlabel('Provider Name')
plt.ylabel('Count')
plt.title('Countplot of Provider')
plt.subplot(2,1,2)
plt.pie(pie_temp,labels=list_temp1,autopct='%1.2f%%')
plt.title('Pie chart of Provider')
# plt.legend()
plt.tight_layout()
plt.show()
#%%
fig = plt.figure(figsize=(24,8))
ax1= fig.add_subplot(1,2,1)
plt.hist(df22.TotalAmount,bins=50)
plt.title('Histogram of Total Amount')
plt.grid()

ax2 = fig.add_subplot(1,2,2)
qqplot(df22.TotalAmount,
       ax=ax2,
       line='s')
plt.grid()
plt.title('QQ-plot of Total Amount')
plt.tight_layout()
plt.show()
#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash('My app', external_stylesheets=external_stylesheets)
my_app.layout = html.Div([
    html.H3('Choose plot type from Tab'),
    dcc.Tabs(id='tabs',value='boxplot',children=[
        dcc.Tab(label='boxplot',value='boxplot'),
        dcc.Tab(label='violin plot',value='violin'),
        dcc.Tab(label='scatter plot',value='scatterplot'),
        # dcc.Tab(label='line plot',value='lineplot'),
        dcc.Tab(label='pie',value='pie'),
        dcc.Tab(label='histogram',value='histogram'),
        dcc.Tab(label='bar plot',value='histogram2'),
        dcc.Tab(label='heatmap',value='heatmap'),
        dcc.Tab(label='scatter matrix',value='matrix'),
        dcc.Tab(label='PCA',value='pca'),
        dcc.Tab(label='Normality Test',value='normaltest'),
    ]),
    html.Div(id='area')
])
@my_app.callback(
    Output('area','children'),
    Input('tabs','value')
)
def updatearea(tabs):
    if tabs=='boxplot':
        return html.Div([
            html.H3('Please choose a feature'),
            dcc.Dropdown(id='drop-boxplot',
                         options=drop_list_boxplot),
            html.Br(),
            html.H3('Remove outliers'),
            dcc.Dropdown(id='drop2-boxplot',value='True',
                         options=[{'label': 'Before removing outliers', 'value': 'False'},
                                  {'label': 'Remove outliers', 'value': 'True'}]),
            dcc.Graph(id='graph-boxplot')
        ])
    if tabs=='scatterplot':
        return html.Div([
            html.H3('To get results quicker, input sample size'),
            dcc.Input(id='inputsize2', value=60000, type='number'),
            html.Div(id='hint2'),
            html.Br(),
            html.H3('Please choose a feature as x'),
            dcc.Dropdown(id='drop-scatterplot',
                         options=drop_list_boxplot),
            html.Br(),
            html.H3('Please choose a feature as y'),
            dcc.Dropdown(id='drop2-scatterplot',
                         options=drop_list_boxplot),
            html.Br(),
            html.H3('Choose a hue'),
            dcc.Dropdown(id='drop3-scatterplot',
                         options=drop_list_category,
                         value='None'),
            dcc.Graph(id='graph-scatterplot')
        ])
    if tabs=='pie':
        return html.Div([
            html.H3('Choose your category'),
            dcc.Dropdown(id='drop-pie',
                         options=drop_list_category),
            html.Br(),
            dcc.Graph(id='graph-pie')
        ])
    if tabs=='histogram':
        return html.Div([
            html.H3('Choose features'),
            dcc.Dropdown(id='type-hist',options=drop_list_all,
                         multi=True
                         ),
            html.Br(),
            html.H3('Choose the number of bins'),
            dcc.Slider(id='bins-hist',min=10,max=90,step=10,value=50,
                       marks={10:10,50:50,90:90}),
            html.Br(),
            html.H3('Choose direction'),
            dcc.Dropdown(id='way',value='vertical',
                         options=[{'label': 'Horizontal', 'value': 'Horizontal'},
                                           {'label': 'Vertical', 'value': 'vertical'}]),
            dcc.Graph(id='graph-hist')
        ])
    if tabs=='histogram2':
        return html.Div([
            html.H3('Choose sample size'),
            dcc.Slider(id='samplesize',min=1000,max=11000,step=1000,value=5000,
                       marks={1000:1000,6000:6000,11000:11000}),
            html.Br(),
            html.H3('Choose a feature'),
            dcc.Dropdown(id='drop-hist2',options=drop_list_category[0:-1]),
            html.Br(),
            html.H3('Choose a hue'),
            dcc.Dropdown(id='hue-hist2',options=drop_list_category,value='None'),
            html.Br(),
            html.H3('Choose barmode'),
            dcc.Dropdown(id='barmode',value='Stack',options=[
                {'label': 'Stack', 'value': 'Stack'},
                {'label': 'Group', 'value': 'Group'},
            ]),
            html.Br(),
            html.H3('Choose direction'),
            dcc.Dropdown(id='way2', value='vertical',
                         options=[{'label': 'Horizontal', 'value': 'Horizontal'},
                                  {'label': 'Vertical', 'value': 'vertical'}]),
            dcc.Graph(id='barplot'),
            html.H4('Since the dataset is too large, use the whole dataset for barplot will run out of memory')
        ])
    if tabs=='violin':
        return html.Div([
            html.H3('To get results quicker, input sample size'),
            dcc.Input(id='inputsize',value=20000,type='number'),
            html.Div(id='hint'),
            html.Br(),
            html.H3('Choose features'),
            dcc.Dropdown(id='type-violin', options=drop_list_all,
                         ),
            html.Br(),
            html.H3('Show box plot'),
            dcc.Dropdown(id='boxv',value='False',options=[{'label': 'Not show', 'value': 'False'},
                                  {'label': 'show', 'value': 'True'}]),
            html.Br(),
            dcc.Graph(id='graph-violin')
            ])
    if tabs == 'matrix':
        return html.Div([
            html.H3('To get results quicker, input sample size'),
            dcc.Input(id='inputsize1',type='number'),
            html.Div(id='hint1'),
            html.H3('Choose features for scatter matrix'),
            dcc.Dropdown(id='drop-matrix',options=drop_list_boxplot,multi=True,
                         value=['FareAmount','TotalAmount']),
            html.Br(),
            html.H3('Whether to show diagonal line'),
            dcc.Slider(id='slidermatrix',min=0,max=1,step=1,value=1,
                       marks={0:'Not show',1:'Show'}),
            dcc.Graph(id='graph-matrix')
        ])
    if tabs=='heatmap':
        return html.Div([
            html.H3('Here is the heatmap'),
            dcc.Graph(id='heatmap-graph',
                      figure=px.imshow(df22.corr()))
        ])
    if tabs=='pca':
        return html.Div([
            html.H3('Choose method/number to keep features'),
            dcc.Dropdown(id='drop-pca',options=drop_feature_list,value='mle'),
            html.Div(id='div-pca'),
            dcc.Graph(id='graph-pca'),
        ])
    if tabs=='normaltest':
        return html.Div([
            html.H3('Choose type of normality test'),
            dcc.Dropdown(id='drop-norm',options=[
                {'label':'kstest','value':'kstest'},
                {'label':'shapiro test','value':'shapiro'},
                {'label':'da_k_squared test','value':'normaltest'},
            ]),
            html.H3('Choose the feature'),
            dcc.Dropdown(id='feat',options=drop_list_boxplot),
            html.Div(id='output-norm'),
        ])
@my_app.callback(
    Output('output-norm','children'),[Input('drop-norm','value'),Input('feat','value')],
)
def updatenorm(way,feat):
    if way=='kstest':
        result = st.kstest(df22[feat],'norm')
        result_norm = 'Normal' if result[1] > 0.01 else 'Not Normal'
        return html.Div([
            html.H3(f'K-S test: statistics={result[0]}, p-value={result[1]}'),
            html.H3(f'K-S test: {feat} looks {result_norm}')
        ])
    if way=='shapiro':
        result = st.shapiro(df22[feat])
        result_norm = 'Normal' if result[1] > 0.01 else 'Not Normal'
        return html.Div([
            html.H3(f'Shapiro test: statistics={result[0]}, p-value={result[1]}'),
            html.H3(f'Shapiro test: {feat} looks {result_norm}')
        ])
    if way=='normaltest':
        result = st.normaltest(df22[feat])
        result_norm = 'Normal' if result[1] > 0.01 else 'Not Normal'
        return html.Div([
            html.H3(f'da_k_squared test: statistics={result[0]}, p-value={result[1]}'),
            html.H3(f'da_k_squared test: {feat} looks {result_norm}'),
        ])
@my_app.callback(
    [Output('div-pca','children'),Output('graph-pca','figure')],Input('drop-pca','value'),
)
def updatepca(num):
    X = df22[feature_list].values
    Y = df22[target].values
    # PCA test
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=num,
              svd_solver='full')
    pca.fit(X)
    X_PCA = pca.transform(X)
    fig = px.line(x=np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1),y=np.cumsum(pca.explained_variance_ratio_))
    return [html.Div([
        html.H3(f'Original Shape of original X is {X.shape}'),
        html.H3(f'Transformed X has the shape of {X_PCA.shape}'),
        html.H3(f'Explained variance ratio: {pca.explained_variance_ratio_}'),
        html.H3(f'Original X condition number: {LA.cond(X)}'),
        html.H3(f'Transformed X condition number: {LA.cond(X_PCA)}'),
    ]),fig]
@my_app.callback(
    [Output('hint1','children'),Output('graph-matrix','figure')],
    [Input('inputsize1','value'),Input('drop-matrix','value'),Input('slidermatrix','value')]
)
def updatemat(size,fea,show):
    if size>len(df22):
        return [f'The input is larger than the size of data! The length of dataset is {len(df22)}','']
    fig = px.scatter_matrix(df22,
                            dimensions=fea)
    if show==0:
        fig.update_traces(diagonal_visible=False)
    return ['',fig]
@my_app.callback(
    Output('barplot','figure'),
    [Input('drop-hist2','value'),
    Input('hue-hist2','value'),
    Input('barmode','value'),
    Input('way2','value'),
    Input('samplesize','value')]
)
def updatebar(drop,hue,bar,way2,size):
    if hue == 'None':
        if way2=='vertical':
            fig = px.bar(df22.sample(size),
                         x=drop)
            return fig
        else:
            fig = px.bar(df22.sample(size),
                         y=drop)
            return fig
    else:
        if way2=='vertical':
            if bar=='Stack':
                fig = px.bar(df22.sample(size),x=drop,color=hue)
                return fig
            else:
                fig = px.bar(df22.sample(size),x=drop,color=hue,barmode='group')
                return fig
        else:
            if bar=='Stack':
                fig = px.bar(df22.sample(size),y=drop,color=hue)
                return fig
            else:
                fig = px.bar(df22.sample(size),y=drop,color=hue,barmode='group')
                return fig
@my_app.callback(
                Output('graph-hist','figure'),
                [Input('type-hist','value'),
                 Input('bins-hist','value'),
                 Input('way','value')],
)
def updatehist(features,bins,way):
    if way=='Horizontal':
        fig = go.Figure()
        for i in features:
            fig.add_trace(go.Histogram(y=df22[i], nbinsy=bins))
        return fig
    else:
        fig = go.Figure()
        for i in features:
            fig.add_trace(go.Histogram(x=df22[i],nbinsx=bins))
        return fig
@my_app.callback(
                [Output('graph-violin','figure'),Output('hint','children')],
                [Input('type-violin','value'),
                 Input('boxv','value'),
                 Input('inputsize','value')],
)
def updateviolin(features,way,size):
    if size>len(df22):
        return ['',f'The input is larger than the size of data! The length of dataset is {len(df22)}']
    if way=='False':
        fig = px.violin(df22.sample(size),
                           y=features,)
        return [fig,'']
    else:
        fig = px.violin(df22.sample(size),
                           y=features,
                           box=True)
        return [fig,'']
@my_app.callback(
    Output('graph-pie','figure'),
    Input('drop-pie','value')
)
def updatepie(pie):
    fig = px.pie(df22,
                 pie)
    return fig
@my_app.callback(
    Output(component_id='graph-boxplot',
            component_property='figure'),
    [Input(component_id='drop-boxplot',
           component_property='value'),
     Input(component_id='drop2-boxplot',
           component_property='value'),]
)
def updateboxplot(drop_boxplot,drop2_boxplot):
    if drop2_boxplot=='True':
        fig_boxplot=px.box(df22,
                   y=drop_boxplot)
        return fig_boxplot
    if drop2_boxplot == 'False':
        fig_boxplot=px.box(df21,
                   y=drop_boxplot)
        return fig_boxplot
@my_app.callback(
    [Output(component_id='graph-scatterplot',
            component_property='figure'),
     Output(component_id='hint2',
            component_property='children')],
    [Input(component_id='drop-scatterplot',
           component_property='value'),
     Input(component_id='drop2-scatterplot',
           component_property='value'),
     Input(component_id='drop3-scatterplot',
           component_property='value'),
     Input(component_id='inputsize2',
           component_property='value')]
)
def updatescatterplot(drop_scatterplot,drop2_scatterplot,drop3_scatterplot,size):
    if size>len(df22):
        return ['',f'The input is larger than the size of data! The length of dataset is {len(df22)}']
    if drop3_scatterplot=='None':
        fig_scatterplot=px.scatter(df22.sample(size),
                             x=drop_scatterplot,
                             y=drop2_scatterplot)
        return [fig_scatterplot,'']
    else:
        fig_scatterplot=px.scatter(df22.sample(size),
                             x=drop_scatterplot,
                             y=drop2_scatterplot,
                             color=drop3_scatterplot)
        return [fig_scatterplot,'']
my_app.run_server(
    host= '0.0.0.0',
    port=8888
)

