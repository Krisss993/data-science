import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly_express as px
import plotly
import plotly.offline as pyo
import plotly.graph_objects as go


sns.set()


fig = plt.figure()
fig.show()

plt.plot([0,1, 2],[0,3, 2], label='line', color='green')
plt.legend()
plt.xlabel('os X')
plt.ylabel('os Y')
plt.title('sample')
plt.show()

x = np.arange(-3, 3, 0.1)

x_sin = np.sin(x)
x_cos = np.cos(x)

plt.plot(x, x_sin, label='sinus')
plt.plot(x, x_cos, label='cosinus')
plt.legend()
plt.show()

fig = plt.figure(figsize=(12,8))
plt.subplot(121)
plt.plot([0,1, 2],[0,3, 2], label='line', color='green')
plt.subplot(122)
plt.plot(x, x_sin, label='sinus')
plt.plot(x, x_cos, label='cosinus')
fig.show()


plt.bar(x=[0,1,2,3], height=[4,7,1,5])

plt.barh(y=[0,1,2,3], width=[4,7,1,5])

x1 = np.random.randn(300)
x1
x2 = np.random.randn(300)

x3 = np.random.randn(300) * 50
plt.scatter(x=x1,y=x2, s=x3, alpha = 0.5)


loc = r'C:\Users\kpas9\Pictures\XC60 II\2018-Volvo-XC60-1.jpg'

img = mpimg.imread(loc)

img
img.shape
plt.imshow(img)

fig = plt.figure(figsize=(12,10))
plt.subplot(221)
plt.imshow(mpimg.imread(loc))
plt.subplot(222)
plt.imshow(mpimg.imread(r'C:\Users\kpas9\Pictures\XC60 II\2018-Volvo-XC60-2.jpg'))
plt.subplot(223)
plt.imshow(mpimg.imread(r'C:\Users\kpas9\Pictures\XC60 II\2018-Volvo-XC60-3.jpg'))
plt.subplot(224)
plt.imshow(mpimg.imread(r'C:\Users\kpas9\Pictures\XC60 II\2018-Volvo-XC60-4.jpg'))
fig.show()

fig = plt.figure(figsize=(12,10))
for idx in range(1,5):
    plt.subplot(220 + idx)
    plt.imshow(mpimg.imread(r'C:\Users\kpas9\Pictures\XC60 II\2018-Volvo-XC60-{}.jpg'.format(idx)))
plt.show()
    
fig = plt.figure(figsize=(12,10))
for idx in range(1,5):
    plt.subplot(140 + idx)
    plt.imshow(mpimg.imread(r'C:\Users\kpas9\Pictures\XC60 II\2018-Volvo-XC60-{}.jpg'.format(idx)))     
plt.show()

fig = plt.figure(figsize=(12,10))
for idx in range(1,10):
    plt.subplot(330 + idx)
    plt.imshow(mpimg.imread(r'C:\Users\kpas9\Pictures\XC60 II\2018-Volvo-XC60-{}.jpg'.format(idx)))
plt.show()


x = np.arange(-3,3,0.1)
sin_x = np.sin(x)                   

plt.style.available

plt.style.use('dark_background')
plt.plot(x,sin_x)








#  # #  ## # #                   S E A B O R N                        ## # # # # # 













sns.set()
sns.__version__




df = sns.load_dataset('tips')
df

df.info()

df.describe().T

df.describe(include=['category']).T

sns.relplot(data=df, x='total_bill', y='tip')



sns.set(font_scale=1.2)
# hue - rozbija dane(poszczególne kropki) na kategorie - tutaj size
# col - subploty na 2 kolumny
# row - subploty na 2 wiersze
# dostajemy 4 wykresy 2x2
sns.relplot(data=df, x='total_bill', y='tip', size='size', hue='size', col='time', row='smoker')

sns.catplot(data=df, x='day', y='total_bill', kind='swarm')
sns.catplot(data=df, x='day', y='total_bill', kind='box')
sns.catplot(data=df, x='day', y='total_bill', kind='violin')
sns.catplot(data=df, x='day', y='total_bill', kind='bar')

# CZESTOSC WYSTEPOWANIA KAZDEGO DNIA
sns.catplot(data=df, x='day', kind='count')

df = sns.load_dataset('titanic')
df

sns.catplot(data=df, x='deck', kind='count', palette='Blues')

sns.catplot(data=df, y='deck', kind='count', palette='Blues')

sns.pairplot(df)

df = sns.load_dataset('iris')
df

sns.pairplot(df, hue='species')










#  # #  ## # #                  P l o t l y   E x p r e s s                        ## # # # # # 









df = px.data.iris()
df


# NIE DZIAŁA
# pyo.init_notebook_mode(connected=True)
# fig = px.scatter(data_frame=df, x='sepal_length', y='sepal_width')
# fig.show()

# WYKRES PUNKTOWY
fig = px.scatter(data_frame=df, x='sepal_length', y='sepal_width', width=500, height=400)
pyo.plot(fig, filename='plotly_express_plot.html')

# KOLOR W ZALEZNOSCI OD species
fig = px.scatter(data_frame=df, x='sepal_length', y='sepal_width', width=700, height=400, color='species')
pyo.plot(fig, filename='plotly_express_plot.html')

# DODANIE 2 WYKRESOW POBOCZNYCH W TYPIE violin i box
fig = px.scatter(data_frame=df, x='sepal_length', y='sepal_width', width=1000, height=700, color='species', marginal_x='violin', marginal_y='box', title='IRIS')
pyo.plot(fig, filename='plotly_express_plot.html')

# DODANIE LINI TRENDU ols
fig = px.scatter(data_frame=df, x='sepal_length', y='sepal_width', width=1000, height=700, color='species', title='IRIS', trendline='ols')
pyo.plot(fig, filename='plotly_express_plot.html')

df.columns
# WYKRESY KORELACJI 
fig = px.scatter_matrix(data_frame=df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species')
pyo.plot(fig, filename='plot.html')

# LINOWY WYKRES WSPOLZALEZNOSCI
fig = px.parallel_coordinates(data_frame=df, color='species_id')
pyo.plot(fig, filename='plot.html')






df = px.data.tips()
df

df.info()


# FACET COL DZIELI NA ILOSC WYKRESOW W ZALEZNOSCI OD day
# category_orders POZWALA USTALIC KOLEJNOSC PODAJAC JA W SLOWNIKU
fig = px.scatter(data_frame=df, x='total_bill', y='tip', facet_col='day', category_orders={'day':['Thur','Fri','Sat','Sun']}, trendline='ols', color='smoker')
pyo.plot(fig, 'plot.html')

# facet_row DZIELI NA WIERSZE
fig = px.scatter(data_frame=df, x='total_bill', y='tip', facet_row='time', trendline='ols', color='smoker')
pyo.plot(fig, 'plot.html')


# LINOWY WYKRES WSPOLZALEZNOSCI
fig = px.parallel_categories(data_frame=df, color='size')
pyo.plot(fig,'plot.html')


df = px.data.gapminder()
df
df.info()
fig = px.scatter(data_frame=df.query('year == 2007'), x='gdpPercap', y='lifeExp', size='pop', color='continent', log_x=True, hover_name='country', size_max=60)
pyo.plot(fig,'plot.html')

# ANIMACJA Z PODZIALEM NA 4 WYKRESY, KAZDY DOTYCZY WARTOSCI Z continent --> facet_col='continent'
fig = px.scatter(data_frame=df, x='gdpPercap', y='lifeExp', size='pop', color='continent', log_x=True, hover_name='country', size_max=60, 
                 animation_frame='year', range_y=[25,90], facet_col='continent', animation_group='country')
pyo.plot(fig,'plot.html')

# ANIMACJA
fig = px.scatter(data_frame=df, x='gdpPercap', y='lifeExp', size='pop', color='continent', log_x=True, hover_name='country', size_max=60, 
                 animation_frame='year', range_y=[25,90], animation_group='country')
pyo.plot(fig,'plot.html')

# WYKRES LINIOWY
fig = px.line(data_frame=df.query('continent == "Europe"'), x='year', y='pop', color='country')
pyo.plot(fig,'plot.html')


def fetch_financial_data(company='AMZN'):
    import pandas_datareader.data as web
    return web.DataReader(name=company, data_source='stooq')

df_raw = fetch_financial_data()
df = df_raw.copy()
df.reset_index(inplace=True)
df

# WYKRES LINIOWY Z ZAZNACZONĄ POWIERCHNIĄ POD
fig = px.area(data_frame=df, x='Date', y='Close')
pyo.plot(fig,'plot.html')

# WYKRES LINIOWY LOGARYTNIMCZNY Z ZAZNACZONĄ POWIERCHNIĄ POD
fig = px.area(data_frame=df, x='Date', y='Close', log_y=True)
pyo.plot(fig,'plot.html')


df = sns.load_dataset('flights')
df

# WYKRES SŁUPKOWY
fig = px.bar(data_frame=df, x='year',y='passengers', color='year')
pyo.plot(fig, 'plot.html')

# WYKRES SŁUPKOWY HORYZONTALNY
fig = px.bar(data_frame=df, x='passengers',y='year', color='year', orientation='h')
pyo.plot(fig, 'plot.html')

# HISTOGRAM
fig = px.histogram(data_frame=df, x='passengers', nbins=50)
pyo.plot(fig, 'plot.html')



df = pd.read_csv("https://ml-repository-krakers.s3-eu-west-1.amazonaws.com/plotly-course/us-cities-top-1k.csv")
df.head()
# WYŚWIETLENIE MAPY INTERAKTYWNEJ
fig = px.scatter_mapbox(data_frame=df, lat='lat', lon='lon', hover_name='City', hover_data=['State', 'Population'], zoom=3)
fig.update_layout(mapbox_style='carto-positron', margin={'r':10,'t':10,'l':10,'b':10})
pyo.plot(fig, 'plot.html')
























#  # #  ## # #                  P l o t l y                  ## # # # # # 



plotly.__version__












# PROSTY WYKRES KOLUMNOWY
fig = go.Figure(
    data=go.Bar(y=[2,3,1,4])
    )
pyo.plot(fig,'plot.html')



fig = go.Figure(
    data=go.Bar(y=[2,3,1,4]),
    # DODANIE TYTUŁU
    layout=go.Layout(title={'text':'Wykres slupkowy'})
    )
pyo.plot(fig,'plot.html')



fig = go.Figure(
    data=go.Bar(y=[2,3,1,4]),
    # DODANIE TYTUŁU
    layout=go.Layout(title_text='Wykres slupkowy')
    )
pyo.plot(fig,'plot.html')
fig.write_html('plot.html')


df = sns.load_dataset('diamonds')
df
df.info()

dfv = df['cut'].value_counts()
dfv = dfv.reset_index()
dfv

fig = go.Figure(
    #hole DAJE WYKRES PIERSCIENIOWY
    data = go.Pie(labels=dfv['cut'], values=dfv['count'], hole=0.5),
    layout=go.Layout(title_text='Rozklad zmiennej cut')
    )
pyo.plot(fig,'fig.html')







# DIAGRAM SANKEYA
data = [go.Sankey(node=dict(label=['Nonchurn_2018', 'Churn_2018', 'Nonchurn_2019', 'Churn_2019']),
                 link=dict(source=[0, 0, 1, 1], # indeks odpowiadający etykiecie (labels)
                          target=[2, 3, 2, 3],
                          value=[65, 12, 18, 5]))]

fig = go.Figure(data=data, layout=go.Layout(width=800, height=400))
pyo.plot(fig,'fig.html')



data = [go.Sankey(node=dict(label=['Nonchurn_2018', 'Churn_2018', 'Nonchurn_2019', 'Churn_2019','Nonchurn_2020', 'Churn_2020']),
                 link=dict(source=[0, 0, 1, 1, 2, 3], # indeks odpowiadający etykiecie (labels)
                          target=[2, 3, 2, 3, 4, 5],
                          value=[65, 10, 5, 20, 70, 30]))]

fig = go.Figure(data=data, layout=go.Layout(width=1200, height=600))
pyo.plot(fig,'fig.html')






# WYKRES SWIECOWY
df
df.reset_index(inplace=True)
df = df[df['Date']>'2023-01-01']
df

fig = go.Figure(data=go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
pyo.plot(fig,'fig.html')
