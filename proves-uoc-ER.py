import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq

campanyes = pd.read_csv ("D:/Capstone-UOC/Data/dades-capstone-def3_2.csv", parse_dates=True)

campanyes.info()

#validamos las columnas cargadas
campanyes.columns

# cálculo de la edad en años
campanyes['data_naixement']=pd.to_datetime(campanyes['data_naixement'])
campanyes['edad_dias']=(pd.to_datetime(campanyes['data_ini_lead'])-pd.to_datetime(campanyes['data_naixement']))
campanyes['edad_anyos']=campanyes['edad_dias'].astype('timedelta64[D]')/365.24

edad_media_anyos=campanyes['edad_anyos'].mean()
edad_media_anyos

campanyes['edad_anyos'].hist(bins=90) 

edad_media_anyos


# recuento de usuarios por semestre
campanyes.groupby(['semestre'])['identif_usuari'].count()

# recuento de usuarios por semestre
campanyes.groupby(['semestre','estat_lead_recode'])['producte_comprat_recode'].count()

# agrupamos por identif_usuari y calculamos first, promedio, count_distinct para diferentes campos
Persones_Campanya_sexe = pd.DataFrame(campanyes.groupby(['identif_usuari'])['sexe'].first())  #sexo 
Persones_Campanya_edat = pd.DataFrame(campanyes.groupby(['identif_usuari'])['edad_anyos'].mean())  #edad
Persones_Campanya_prodcomp = pd.DataFrame(campanyes.groupby(['identif_usuari'])['producte_comprat_recode'].nunique())  #producto comprado
Persones_Campanya_puntentr = pd.DataFrame(campanyes.groupby(['identif_usuari'])['punt_entrada_recode'].nunique())  #punto entrada
Persones_Campanya_area = pd.DataFrame(campanyes.groupby(['identif_usuari'])['area_prod_comprat_recode'].nunique())  #área
Persones_Campanya_subarea = pd.DataFrame(campanyes.groupby(['identif_usuari'])['subarea_prod_comprat_recode'].nunique())  #subarea
Persones_Campanya_tipusprod = pd.DataFrame(campanyes.groupby(['identif_usuari'])['tipus_producte_comprat'].nunique())  #tipo de producto
Persones_Campanya_canal_recode = pd.DataFrame(campanyes.groupby(['identif_usuari'])['canal_recode'].nunique())  #canal
Persones_Campanya_idioma_recode = pd.DataFrame(campanyes.groupby(['identif_usuari'])['idioma_recode'].nunique())  #idioma
Persones_Campanya_semestre = pd.DataFrame(campanyes.groupby(['identif_usuari'])['semestre'].nunique())  #semestre
Persones_Campanya_regio = pd.DataFrame(campanyes.groupby(['identif_usuari'])['regio'].nunique())  #región

#añadimos índice para liberar el código de usuario que está siendo utilizado previamente como clave
Persones_Campanya_sexe = Persones_Campanya_sexe.reset_index(drop=False)
Persones_Campanya_edat = Persones_Campanya_edat.reset_index(drop=False)
Persones_Campanya_prodcomp = Persones_Campanya_prodcomp.reset_index(drop=False)
Persones_Campanya_puntentr = Persones_Campanya_puntentr.reset_index(drop=False)
Persones_Campanya_area = Persones_Campanya_area.reset_index(drop=False)
Persones_Campanya_subarea = Persones_Campanya_subarea.reset_index(drop=False)
Persones_Campanya_tipusprod = Persones_Campanya_tipusprod.reset_index(drop=False)
Persones_Campanya_canal_recode = Persones_Campanya_canal_recode.reset_index(drop=False)
Persones_Campanya_idioma_recode = Persones_Campanya_idioma_recode.reset_index(drop=False)
Persones_Campanya_semestre = Persones_Campanya_semestre.reset_index(drop=False)
Persones_Campanya_regio = Persones_Campanya_regio.reset_index(drop=False)

#unimos los dataframes con información por usuario a una tabla única
Persones_activitat = pd.merge(Persones_Campanya_sexe, Persones_Campanya_edat, on='identif_usuari')
Persones_activitat = pd.merge(Persones_activitat, Persones_Campanya_prodcomp, on='identif_usuari')
Persones_activitat = pd.merge(Persones_activitat, Persones_Campanya_puntentr, on='identif_usuari')
Persones_activitat = pd.merge(Persones_activitat, Persones_Campanya_area, on='identif_usuari')
Persones_activitat = pd.merge(Persones_activitat, Persones_Campanya_subarea, on='identif_usuari')
Persones_activitat = pd.merge(Persones_activitat, Persones_Campanya_tipusprod, on='identif_usuari')
Persones_activitat = pd.merge(Persones_activitat, Persones_Campanya_canal_recode, on='identif_usuari')
Persones_activitat = pd.merge(Persones_activitat, Persones_Campanya_idioma_recode, on='identif_usuari')
Persones_activitat = pd.merge(Persones_activitat, Persones_Campanya_regio, on='identif_usuari')

Persones_activitat.describe() #descriptivas de los índices obtenidos

#Borramos las tablas intermedias
del Persones_Campanya_sexe
del Persones_Campanya_edat
del Persones_Campanya_prodcomp
del Persones_Campanya_puntentr
del Persones_Campanya_area
del Persones_Campanya_subarea
del Persones_Campanya_tipusprod
del Persones_Campanya_canal_recode
del Persones_Campanya_idioma_recode
del Persones_Campanya_semestre
del Persones_Campanya_regio


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

evol_campanya_20151

evol_campanya_20151.plot()
evol_campanya_20152.plot()
evol_campanya_20161.plot()




# clustering
data = Persones_activitat.ix[:,3:11]

# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,10)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()






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




