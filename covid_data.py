#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# print(data), the time_series csv file describes each country's situations until 2020-04-10
csv = 'time-series-19-covid-combined.csv'
data = pd.read_csv(csv, error_bad_lines=False)


# In[3]:


#Here we list the content of each column
list(data)


# In[4]:


#this dataset shows the total numbers of confirmed,recovered and death til the latest date(2020-04-17) in each country
lastDate = data['Date'].max()
df_sum = data.loc[data['Date'] == lastDate]
print(df_sum)


# In[5]:


data_latest = df_sum.groupby(['Country/Region'],as_index=False).agg({'Confirmed':sum, 'Recovered':sum, 'Deaths':sum})

#now we select a subset of countries with highest number of confirmed cases
data_10 = data_latest.sort_values('Confirmed', ascending=False)[:10]
print(data_10)


# In[6]:


#extract the top 10 countries's time series data  ['US', 'Spain', 'Italy', 'France', 'Germany', 'China','United Kingdom','Iran', 'Turkey', 'Belgium']
data0 = data.loc[data['Country/Region'].isin(data_10['Country/Region'].tolist())]

#For convenience, we seperate each country's data
data_us  = data0.loc[data['Country/Region'] == 'US']
data_sp  = data0.loc[data['Country/Region'] == 'Spain']
data_it = data0.loc[data['Country/Region'] == 'Italy']
data_fr =  data0.loc[data['Country/Region'] == 'France']
data_ge = data0.loc[data['Country/Region'] == 'Germany']
data_cn = data0.loc[data['Country/Region'] == 'China']
data_uk = data0.loc[data['Country/Region'] == 'United Kingdom']
data_ir = data0.loc[data['Country/Region'] == 'Iran']
data_tk = data0.loc[data['Country/Region'] == 'Turkey']
data_bl = data0.loc[data['Country/Region'] == 'Belgium']


# In[7]:


#check which countries have multiple regions, use group_by() to add
print(len(data_us))
print(len(data_sp))
print(len(data_it))
print(len(data_fr)) #has
print(len(data_ge))
print(len(data_cn)) #has
print(len(data_uk)) #has
print(len(data_ir))
print(len(data_tk))
print(len(data_bl))


# In[8]:


data_fr1 = data_fr.groupby(['Date']).agg({'Confirmed':sum, 'Recovered':sum, 'Deaths':sum})
data_cn1 = data_cn.groupby(['Date']).agg({'Confirmed':sum, 'Recovered':sum, 'Deaths':sum})
data_uk1 = data_uk.groupby(['Date']).agg({'Confirmed':sum, 'Recovered':sum, 'Deaths':sum})


# In[9]:


#since we need fit the function, we need create a column of the day(the first, second, third...)
data_us.loc[:,'Day'] = range(1,len(data_us)+1)
data_cn1.loc[:,'Day'] = range(1,len(data_cn1)+1)
data_sp.loc[:,'Day'] = range(1,len(data_sp)+1)
data_it.loc[:,'Day'] = range(1,len(data_it)+1)
data_fr1.loc[:,'Day'] = range(1,len(data_fr1)+1)
data_ge.loc[:,'Day'] = range(1,len(data_ge)+1)
data_uk1.loc[:,'Day'] = range(1,len(data_uk1)+1)
data_ir.loc[:,'Day'] = range(1,len(data_ir)+1)
data_tk.loc[:,'Day'] = range(1,len(data_tk)+1)
data_bl.loc[:,'Day'] = range(1,len(data_bl)+1)


# In[10]:


#a)
#Here we define two functions: Exponential function and Logistic function.
#Then we use this two functions to fit regarding different countries
def func1(t,a,k,t0):
    return a*np.exp(k*(t-t0))

def func2(t,L,k,t0):
    return L/(1+np.exp(-k*(t-t0)))


# In[11]:


from scipy.optimize import curve_fit

t = data_us['Day']
t = np.array(t)
y = data_us['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func1,t,y,maxfev = 30000)
print(popt)
a = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func1(t,a,k,t0) 


# In[12]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('US confirmed cases curve_fit')
plt.show()


# In[13]:


t = data_cn1['Day']
t = np.array(t)
y = data_cn1['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func2,t,y,maxfev = 30000)
print(popt)
L = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func2(t,L,k,t0)


# In[14]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('China confirmed cases curve_fit')
plt.show()


# In[15]:


t = data_it['Day']
t = np.array(t)
y = data_it['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func1,t,y,maxfev = 30000)
print(popt)
a = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func1(t,a,k,t0)


# In[16]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('Italy confirmed cases curve_fit')
plt.show()


# In[17]:


t = data_sp['Day']
t = np.array(t)
y = data_sp['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func1,t,y,maxfev = 30000)
print(popt)
a = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func1(t,a,k,t0)


# In[18]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('Spain confirmed cases curve_fit')
plt.show()


# In[19]:


t = data_fr1['Day']
t = np.array(t)
y = data_fr1['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func1,t,y,maxfev = 30000)
print(popt)
a = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func1(t,a,k,t0)


# In[20]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.ylabel('y')
plt.xlabel('t')
plt.legend()
plt.title('France confirmed cases curve_fit')
plt.show()


# In[21]:


t = data_ge['Day']
t = np.array(t)
y = data_ge['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func1,t,y,maxfev = 30000)
print(popt)
a = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func1(t,a,k,t0)


# In[22]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.ylabel('y')
plt.xlabel('t')
plt.legend()
plt.title('German confirmed cases curve_fit')
plt.show()


# In[23]:


t = data_uk1['Day']
t = np.array(t)
y = data_uk1['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func1,t,y,maxfev = 30000)
print(popt)
a = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func1(t,a,k,t0)


# In[24]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.ylabel('y')
plt.xlabel('t')
plt.legend()
plt.title('UK confirmed cases curve_fit')
plt.show()


# In[25]:


t = data_ir['Day']
t = np.array(t)
y = data_ir['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func1,t,y,maxfev = 30000)
print(popt)
a = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func1(t,a,k,t0)


# In[26]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.ylabel('y')
plt.xlabel('t')
plt.legend()
plt.title('Iran confirmed cases curve_fit')
plt.show()


# In[27]:


t = data_tk['Day']
t = np.array(t)
y = data_tk['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func1,t,y,maxfev = 30000)
print(popt)
a = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func1(t,a,k,t0)


# In[28]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.ylabel('y')
plt.xlabel('t')
plt.legend()
plt.title('Turkey confirmed cases curve_fit')
plt.show()


# In[29]:


t = data_bl['Day']
t = np.array(t)
y = data_bl['Confirmed']
y = np.array(y)

popt, pcov = curve_fit(func1,t,y,maxfev = 30000)
print(popt)
a = popt[0] 
k = popt[1]
t0 = popt[2]
yvals = func1(t,a,k,t0)


# In[30]:


plot1 = plt.plot(t, y, 'o', label='real values')
plot2 = plt.plot(t, yvals,'r', label='polyfit values')
plt.ylabel('y')
plt.xlabel('t')
plt.legend()
plt.title('Belgium confirmed cases curve_fit')
plt.show()


# In[66]:


#create a plot contains all countries'real growth curves
plt.figure(figsize=(10,8))
plt.plot(t, data_us['Confirmed'], label='US')
plt.plot(t, data_cn1['Confirmed'], label='China')
plt.plot(t, data_fr1['Confirmed'], label='France')
plt.plot(t, data_ge['Confirmed'], label='German')
plt.plot(t, data_it['Confirmed'], label='Italy')
plt.plot(t, data_uk1['Confirmed'], label='UK')
plt.plot(t, data_tk['Confirmed'], label='Turkey')
plt.plot(t, data_sp['Confirmed'], label='Spain')
plt.plot(t, data_ir['Confirmed'], label='Iran')
plt.plot(t, data_bl['Confirmed'], label='Belgium')
plt.xlabel('t')
plt.ylabel('Confirmed Case')
plt.legend()
plt.title('Growth of Confirmed cases in top 10 countries')
plt.show()


# In[31]:


#b)
#Compare the statistics conﬁrmed cases, intend to create box-plot to analyze
#Here we need to exlarge the subset because 10 data points subset is too small
data_add = data_latest.sort_values('Confirmed', ascending=False)[10:20]
data_add


# In[38]:


data1 = data.loc[data['Country/Region'].isin(data_add['Country/Region'].tolist())]
data_sl =  data1.loc[data['Country/Region'] == 'Switzerland']
data_nl = data1.loc[data['Country/Region'] == 'Netherlands']
data_ca  = data1.loc[data['Country/Region'] == 'Canada']
data_br = data1.loc[data['Country/Region'] == 'Brazil']
data_pt = data1.loc[data['Country/Region'] == 'Portugal']
data_au = data1.loc[data['Country/Region'] == 'Austria']
data_ru  = data1.loc[data['Country/Region'] == 'Russia']
data_kr = data1.loc[data['Country/Region'] == 'Korea, South']
data_is = data1.loc[data['Country/Region'] == 'Israel']
data_sd = data1.loc[data['Country/Region'] == 'Sweden']


# In[39]:


print(len(data_ca))#has
print(len(data_ru))
print(len(data_nl))#has
print(len(data_sl)) 
print(len(data_pt))
print(len(data_au)) 
print(len(data_kr)) 
print(len(data_is))
print(len(data_sd))
print(len(data_br))


# In[40]:


data_ca1 = data_ca.groupby(['Date']).agg({'Confirmed':sum, 'Recovered':sum, 'Deaths':sum})
data_nl1 = data_nl.groupby(['Date']).agg({'Confirmed':sum, 'Recovered':sum, 'Deaths':sum})


# In[41]:


confirmed_np = np.vstack((data_us['Confirmed'],data_sp['Confirmed'],data_it['Confirmed'],data_fr1['Confirmed'],
                         data_ge['Confirmed'],data_cn1['Confirmed'],data_uk1['Confirmed'],data_ir['Confirmed'],
                         data_tk['Confirmed'],data_bl['Confirmed'],data_sl['Confirmed'],data_nl1['Confirmed'],
                         data_ca1['Confirmed'],data_br['Confirmed'],data_pt['Confirmed'],data_au['Confirmed'],
                         data_ru['Confirmed'],data_kr['Confirmed'],data_is['Confirmed'],data_sd['Confirmed'],
                        )).transpose()


# In[42]:


confirmed_data = pd.DataFrame(confirmed_np,
                                     columns=['US', 'Spain', 'Italy', 'France', 'Germany', 'China', 'UK', 'Iran',
                                              'Turkey', 'Belgium', 'Switzerland', 'Netherlands', 'Canada', 'Brazil', 'Portugal',
                                            'Austria', 'Russia', 'Korea, South', 'Israel','Sweden' ])


# In[43]:


confirmed_data


# In[45]:


#Get the confirmed conlumn of each country,and create a list.
confirmed = confirmed_data.loc[79].tolist()
df_confirmed = pd.DataFrame(confirmed)
print(df_confirmed.describe())


# In[59]:


#For the 20 countries, we can know that:
Q1 = 14992
median = 25609
Q3 = 92748
IQR = Q3-Q1
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper)
print(lower)


# In[60]:


df_confirmed.plot.box(title="Confirmed Cases in the top 20 countries", showmeans=True)
plt.grid(linestyle="--", alpha=0.3)
plt.show() #from the box plot, we can see there is an outlier,which is number of US's cases


# In[48]:


#Get the growth number of each country everyday compared to the previous day
#Since we need to graph multiple boxplots to compare, we still use 10 countries.
#US
x0 = confirmed_data.US[:-1]
x1 = confirmed_data.US[1:]
us_increase = np.array(x1)-np.array(x0)
#Spain
x0 = confirmed_data.Spain[:-1]
x1 = confirmed_data.Spain[1:]
sp_increase = np.array(x1)-np.array(x0)
#Italy
x0 = confirmed_data.Italy[:-1]
x1 = confirmed_data.Italy[1:]
it_increase = np.array(x1)-np.array(x0)
#France
x0 = confirmed_data.France[:-1]
x1 = confirmed_data.France[1:]
fr_increase = np.array(x1)-np.array(x0)
#Germany
x0 = confirmed_data.Germany[:-1]
x1 = confirmed_data.Germany[1:]
ge_increase = np.array(x1)-np.array(x0)
#China
x0 = confirmed_data.China[:-1]
x1 = confirmed_data.China[1:]
cn_increase = np.array(x1)-np.array(x0)
#UK
x0 = confirmed_data.UK[:-1]
x1 = confirmed_data.UK[1:]
uk_increase = np.array(x1)-np.array(x0)
#Iran
x0 = confirmed_data.Iran[:-1]
x1 = confirmed_data.Iran[1:]
ir_increase = np.array(x1)-np.array(x0)
#Turkey
x0 = confirmed_data.Turkey[:-1]
x1 = confirmed_data.Turkey[1:]
tk_increase = np.array(x1)-np.array(x0)
#Belgium
x0 = confirmed_data.Belgium[:-1]
x1 = confirmed_data.Belgium[1:]
bl_increase = np.array(x1)-np.array(x0)


# In[49]:


increase_np = np.vstack((us_increase,sp_increase,it_increase,fr_increase,ge_increase,cn_increase,uk_increase,ir_increase,
                         tk_increase,bl_increase)).transpose()


# In[62]:


confirmed_increase = pd.DataFrame(increase_np,columns=['US','Spain','Italy','France','Germany','China','UK','Iran','Turkey','Belgium'])


# In[63]:


#Box plots of everyday growth compared to the previous day of ten countries
plt.figure(figsize=(20,20))
box = confirmed_increase.boxplot(sym='o', 
                   vert=True,  
                   whis=1.5,  
                   patch_artist=True,  
                   meanline=True, showmeans=True,  
                   showbox=True, 
                   showfliers=True,  
                   notch=False,  
                   return_type='dict')

plt.title('10 countries Box plots of everyday growth compared to the previous day')
plt.show()


# In[ ]:




