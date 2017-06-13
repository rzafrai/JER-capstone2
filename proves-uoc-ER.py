
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse

campanyes = pd.read_csv ("D:/Capstone-UOC/Data/dades-capstone-def3_2.csv", parse_dates=True)

campanyes.info()

#validamos las columnas cargadas
campanyes.columns


# cálculo de la edad en años
campanyes['data_naixement']=pd.to_datetime(campanyes['data_naixement'])
campanyes['edad_dias']=(pd.to_datetime(datetime.now())-pd.to_datetime(campanyes['data_naixement']))
campanyes['edad_anyos']=campanyes['edad_dias'].astype('timedelta64[D]')/365.24

edad_media_anyos=campanyes['edad_anyos'].mean()
edad_media_anyos

campanyes['edad_anyos'].hist(bins=90) 

edad_media_anyos


# recuento de usuarios por semestre
campanyes.groupby(['semestre'])['sexe'].count()

# recuento de usuarios por semestre
campanyes.groupby(['semestre','estat_lead_recode'])['producte_comprat_recode'].count()

# agrupamos por identif_usuari y calculamos count_distinct para diferentes campos
Persones_Campanya_sexe = campanyes.groupby(['identif_usuari'])['sexe'].first()  #sexo 
Persones_Campanya_edat = campanyes.groupby(['identif_usuari'])['edad_anyos'].mean()  #edad
Persones_Campanya_prodcomp = campanyes.groupby(['identif_usuari'])['producte_comprat_recode'].nunique()  #producto comprado
Persones_Campanya_puntentr = campanyes.groupby(['identif_usuari'])['punt_entrada_recode'].nunique()  #punto entrada
Persones_Campanya_area = campanyes.groupby(['identif_usuari'])['area_prod_comprat_recode'].nunique()  #área
Persones_Campanya_subarea = campanyes.groupby(['identif_usuari'])['subarea_prod_comprat_recode'].nunique()  #subarea
Persones_Campanya_tipusprod = campanyes.groupby(['identif_usuari'])['tipus_producte_comprat'].nunique()  #tipo de producto
Persones_Campanya_canal_recode = campanyes.groupby(['identif_usuari'])['canal_recode'].nunique()  #canal
Persones_Campanya_idioma_recode = campanyes.groupby(['identif_usuari'])['idioma_recode'].nunique()  #idioma
Persones_Campanya_semestre = campanyes.groupby(['identif_usuari'])['semestre'].nunique()  #semestre
Persones_Campanya_regio = campanyes.groupby(['identif_usuari'])['regio'].nunique()  #región



# filtramos campaña 20151, 20152 y 20161
campanya_20151 = campanyes[campanyes['semestre'] == 20151]
campanya_20152 = campanyes[campanyes['semestre'] == 20152]
campanya_20161 = campanyes[campanyes['semestre'] == 20161]
campanya_20151
campanya_20152
campanya_20161

campanya_20151['data_ini_lead_date']=pd.to_datetime(campanya_20151['data_ini_lead'])
campanya_20152['data_ini_lead_date']=pd.to_datetime(campanya_20152['data_ini_lead'])
campanya_20161['data_ini_lead_date']=pd.to_datetime(campanya_20161['data_ini_lead'])

#graficamos la evolución de leads campanya 20151, 20152 y 20161 por fecha
evol_campanya_20151=campanya_20151.groupby(['data_ini_lead_date'])['producte_comprat_recode'].count()
evol_campanya_20152=campanya_20152.groupby(['data_ini_lead_date'])['producte_comprat_recode'].count()
evol_campanya_20161=campanya_20161.groupby(['data_ini_lead_date'])['producte_comprat_recode'].count()

evol_campanya_20151.plot()
evol_campanya_20152.plot()
evol_campanya_20161.plot()



# In[32]:

group=matricula[["producte_comprat_recode","data_matricula"-"data_naixement"]].groupby('producte_comprat_recode').mean()
group.sort_values(by='data_matricula'-"data_naixement", ascending= False)


# In[2]:

matricula["dias_matricula"] = matricula['data_matricula']-matricula['data_ini_lead']
matricula.tail()


# In[42]:

matricula["dias_inici_fi_campanya"] = matricula['data_fi_campanya']-matricula['data_ini_lead']
matricula.tail()


# In[43]:

matricula["dias_acces"] = matricula['data_acces']-matricula['data_ini_lead']
matricula.tail()


# In[1]:

filtered_data = matricula[matricula["dias_matricula"]>0]  
pivmatricula=pd.pivot_table(filtered_data, values='dias_matricula', index=['producte_comprat_recode'],columns = ['regio'])
pivmatricula


# In[52]:

pivmatricula.rank(ascending=True,method='first')


# In[61]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pylab as plt

fig = plt.figure(figsize=(20,20))
totalSum=pivmatricula.sum(axis=1).sort_values(ascending=False)
totalSum.plot(kind='bar',style='a', alpha=0.4,title = "Dias hasta matricula por Programa")
#plt.savefig("Totalvalue_Country.png",dpi= 300, bbox_inches='tight')
plt.show()


# In[ ]:




