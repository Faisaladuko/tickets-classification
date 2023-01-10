import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re
import matplotlib.pyplot as plt

# Reading data from excel
jan = pd.read_excel('data\Jan.xlsx')
feb = pd.read_excel('data\Feb.xlsx')
mar = pd.read_excel('data\Mar.xlsx')
april = pd.read_excel('data\April.xlsx')
may = pd.read_excel('data\May.xlsx')
jun = pd.read_excel('data\Jun.xlsx')
jul = pd.read_excel('data\Jul.xlsx')
aug = pd.read_excel('data\Aug.xlsx')
sept = pd.read_excel('data\Sept.xlsx')
oct = pd.read_excel('data\Oct.xlsx')

# Joining and preprocessing/Cleaning data
dataa = pd.concat([jan,feb,mar,april,may,jun,jul,aug,sept,oct]).reset_index()
dataa['Description2'] = dataa['Description'].apply(lambda x: str(x).split(':')[1:])
dataa['Description2'] = dataa['Description2'].apply(lambda x: ''.join([str(a) for a in x]))
dataa['Description2'] = dataa['Description2'].apply(lambda x:re.sub(r'[^a-zA-Z0-9 ]'," ",x)) 
dataa['Description2'] = dataa['Description2'].apply(lambda x:re.sub(r'[0-9]',"",x)) 

# Vectorizes text data
vecc = TfidfVectorizer(stop_words = 'english', encoding ='utf-8')
d_fet = vecc.fit_transform(dataa['Description2'])

#Finding the optimal number of clusters for K-Mean
N= range(2, 500, 20)
pdata = pd.DataFrame(columns = ["N","SILH"])
for cluster in N:
    mdl = KMeans(n_clusters = cluster, n_init='auto', random_state = 10)
    labless = mdl.fit_predict(d_fet)
    av_silh_score = silhouette_score(d_fet,labless)
    #print('For N= ',cluster,': Average Silhoute Score = ', av_silh_score)
    newv = {"N":cluster, "SILH":av_silh_score}
    pdata=pdata.append(newv, ignore_index = True)
pdata.plot(kind= 'line', x = 'N', y='SILH')
plt.show

# Fitting K-Mean
mdl2 = KMeans(n_clusters =21,init = 'k-means++',max_iter=200, n_init='auto', random_state = 10)
mdl2 = mdl2.fit(d_fet)
dataa['MyCategories'] = mdl2.labels_

# Extracting features text for each category
words = vecc.get_feature_names_out()
cent = mdl2.cluster_centers_.argsort()[:, ::-1]
xx=[]
wc = pd.DataFrame(columns=['Category', 'KeyWords'])
for f in range(21):
    #print('Catergory: ',f)
    xx.append(f)
    for i in cent[f, :7]:
       w =words[i]
       #print(words[i])
       xx.append(w)
       wnew = {'Category':f, 'KeyWords':w}
       wc = wc.append(wnew,ignore_index=True)
dataa.to_csv('inspect.csv')
assss = pd.pivot(wc, columns=['Category'], values=['KeyWords'])
print(assss)
