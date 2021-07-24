medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from urllib.request import urlretrieve 
#import pdb;pdb.set_trace() 
urlretrieve(medical_charges_url , 'medical.csv')  
import pandas as pd           
medical_df = pd.read_csv('medical.csv')    #reading pandas 
medical_df                  #displaying medical 
medical_df.info()           #gives out information about medical file(dataset)
print(medical_df.describe())        #gives out stats about medical file(dataset)

# matplotlib , seaborn , plotly --> libraries for data visualisation i.e. graphs and plots

import plotly.express as px 
import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sm 
#%matplotlib inline     # ensures charts are not created as pop up but as output in jupyter notebook , so that we dont lose it after closing the pop up

