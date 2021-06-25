# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:26:22 2021

@author: yashesh
"""
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


zomato_real=pd.read_csv("zomato.csv")
'''
Now the next step is data cleaning and feature engineering for this step we need to do a lot of stuff with the data such as:

Deleting Unnecessary Columns
Removing the Duplicates
Remove the NaN values from the dataset
Changing the column names
Data Transformations
Data Cleaning
Adjust the column names
'''
#Deleting Unnnecessary Columns
zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1) #Dropping the column "dish_liked", "phone", "url" and saving the new dataset as "zomato"

#Removing the Duplicates
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)

#Remove the NaN values from the dataset
zomato.isnull().sum()
zomato.dropna(how='any',inplace=True)

#Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})

#Some Transformations
zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].astype(float)

#Removing '/5' from Rates
zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')

# Adjust the column names
zomato.name = zomato.name.apply(lambda x:x.title())
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)

## Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()
    

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

#"[('Rated 4.0', 'RATED\n  A beautiful place to dine in.The interiors take you back to the Mughal era. The lightings are just perfect.We went there on the occasion of Christmas and so they had only limited items available. But the taste and service was not compromised at all.The only complaint is that the breads could have been better.Would surely like to come here again.'), ('Rated 4.0', 'RATED\n  I was here for dinner with my family on a weekday. The restaurant was completely empty. Ambience is good with some good old hindi music. Seating arrangement are good too. We ordered masala papad, panner and baby corn starters, lemon and corrionder soup, butter roti, olive and chilli paratha. Food was fresh and good, service is good too. Good for family hangout.\nCheers'), ('Rated 2.0', 'RATED\n  Its a restaurant near to Banashankari BDA. Me along with few of my office friends visited to have buffet but unfortunately they only provide veg buffet. On inquiring they said this place is mostly visited by vegetarians. Anyways we ordered ala carte items which took ages to come. Food was ok ok. Definitely not visiting anymore.'), ('Rated 4.0', 'RATED\n  We went here on a weekend and one of us had the buffet while two of us took Ala Carte. Firstly the ambience and service of this place is great! The buffet had a lot of items and the good was good. We had a Pumpkin Halwa intm the dessert which was amazing. Must try! The kulchas are great here. Cheers!'), ('Rated 5.0', 'RATED\n  The best thing about the place is itÃ\x83Ã\x83Ã\x82Ã\x82Ã\x83Ã\x82Ã\x82Ã\x92s ambiance. Second best thing was yummy ? food. We try buffet and buffet food was not disappointed us.\nTest ?. ?? ?? ?? ?? ??\nQuality ?. ??????????.\nService: Staff was very professional and friendly.\n\nOverall experience was excellent.\n\nsubirmajumder85.wixsite.com'), ('Rated 5.0', 'RATED\n  Great food and pleasant ambience. Expensive but Coll place to chill and relax......\n\nService is really very very good and friendly staff...\n\nFood : 5/5\nService : 5/5\nAmbience :5/5\nOverall :5/5'), ('Rated 4.0', 'RATED\n  Good ambience with tasty food.\nCheese chilli paratha with Bhutta palak methi curry is a good combo.\nLemon Chicken in the starters is a must try item.\nEgg fried rice was also quite tasty.\nIn the mocktails, recommend ""Alice in Junoon"". Do not miss it.'), ('Rated 4.0', 'RATED\n  You canÃ\x83Ã\x83Ã\x82Ã\x82Ã\x83Ã\x82Ã\x82Ã\x92t go wrong with Jalsa. Never been a fan of their buffet and thus always order alacarteÃ\x83Ã\x83Ã\x82Ã\x82Ã\x83Ã\x82Ã\x82Ã\x92. Service at times can be on the slower side but food is worth the wait.'), ('Rated 5.0', 'RATED\n  Overdelighted by the service and food provided at this place. A royal and ethnic atmosphere builds a strong essence of being in India and also the quality and taste of food is truly authentic. I would totally recommend to visit this place once.'), ('Rated 4.0', 'RATED\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\n\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome'), ('Rated 4.0', 'RATED\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\n\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome'), ('Rated 4.0', 'RATED\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\n\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome')]"

#[('Rated 4.0', 'RATED\n  A beautiful place to dine in.The interiors take you back to the Mughal era. The lightings are just perfect.We went there on the occasion of Christmas and so they had only limited items available. But the taste and service was not compromised at all.The only complaint is that the breads could have been better.Would surely like to come here again.'), ('Rated 4.0', 'RATED\n  I was here for dinner with my family on a weekday. The restaurant was completely empty. Ambience is good with some good old hindi music. Seating arrangement are good too. We ordered masala papad, panner and baby corn starters, lemon and corrionder soup, butter roti, olive and chilli paratha. Food was fresh and good, service is good too. Good for family hangout.\nCheers'), ('Rated 2.0', 'RATED\n  Its a restaurant near to Banashankari BDA. Me along with few of my office friends visited to have buffet but unfortunately they only provide veg buffet. On inquiring they said this place is mostly visited by vegetarians. Anyways we ordered ala carte items which took ages to come. Food was ok ok. Definitely not visiting anymore.'), ('Rated 4.0', 'RATED\n  We went here on a weekend and one of us had the buffet while two of us took Ala Carte. Firstly the ambience and service of this place is great! The buffet had a lot of items and the good was good. We had a Pumpkin Halwa intm the dessert which was amazing. Must try! The kulchas are great here. Cheers!'), ('Rated 5.0', 'RATED\n  The best thing about the place is itÃ\x83Ã\x83Ã\x82Ã\x82Ã\x83Ã\x82Ã\x82Ã\x92s ambiance. Second best thing was yummy ? food. We try buffet and buffet food was not disappointed us.\nTest ?. ?? ?? ?? ?? ??\nQuality ?. ??????????.\nService: Staff was very professional and friendly.\n\nOverall experience was excellent.\n\nsubirmajumder85.wixsite.com'), ('Rated 5.0', 'RATED\n  Great food and pleasant ambience. Expensive but Coll place to chill and relax......\n\nService is really very very good and friendly staff...\n\nFood : 5/5\nService : 5/5\nAmbience :5/5\nOverall :5/5'), ('Rated 4.0', 'RATED\n  Good ambience with tasty food.\nCheese chilli paratha with Bhutta palak methi curry is a good combo.\nLemon Chicken in the starters is a must try item.\nEgg fried rice was also quite tasty.\nIn the mocktails, recommend "Alice in Junoon". Do not miss it.'), ('Rated 4.0', 'RATED\n  You canÃ\x83Ã\x83Ã\x82Ã\x82Ã\x83Ã\x82Ã\x82Ã\x92t go wrong with Jalsa. Never been a fan of their buffet and thus always order alacarteÃ\x83Ã\x83Ã\x82Ã\x82Ã\x83Ã\x82Ã\x82Ã\x92. Service at times can be on the slower side but food is worth the wait.'), ('Rated 5.0', 'RATED\n  Overdelighted by the service and food provided at this place. A royal and ethnic atmosphere builds a strong essence of being in India and also the quality and taste of food is truly authentic. I would totally recommend to visit this place once.'), ('Rated 4.0', 'RATED\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\n\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome'), ('Rated 4.0', 'RATED\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\n\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome'), ('Rated 4.0', 'RATED\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\n\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome')]
#"[('rated 4.0', 'rated\n  a beautiful place to dine in.the interiors take you back to the mughal era. the lightings are just perfect.we went there on the occasion of christmas and so they had only limited items available. but the taste and service was not compromised at all.the only complaint is that the breads could have been better.would surely like to come here again.'), ('rated 4.0', 'rated\n  i was here for dinner with my family on a weekday. the restaurant was completely empty. ambience is good with some good old hindi music. seating arrangement are good too. we ordered masala papad, panner and baby corn starters, lemon and corrionder soup, butter roti, olive and chilli paratha. food was fresh and good, service is good too. good for family hangout.\ncheers'), ('rated 2.0', 'rated\n  its a restaurant near to banashankari bda. me along with few of my office friends visited to have buffet but unfortunately they only provide veg buffet. on inquiring they said this place is mostly visited by vegetarians. anyways we ordered ala carte items which took ages to come. food was ok ok. definitely not visiting anymore.'), ('rated 4.0', 'rated\n  we went here on a weekend and one of us had the buffet while two of us took ala carte. firstly the ambience and service of this place is great! the buffet had a lot of items and the good was good. we had a pumpkin halwa intm the dessert which was amazing. must try! the kulchas are great here. cheers!'), ('rated 5.0', 'rated\n  the best thing about the place is itã\x83ã\x83ã\x82ã\x82ã\x83ã\x82ã\x82ã\x92s ambiance. second best thing was yummy ? food. we try buffet and buffet food was not disappointed us.\ntest ?. ?? ?? ?? ?? ??\nquality ?. ??????????.\nservice: staff was very professional and friendly.\n\noverall experience was excellent.\n\nsubirmajumder85.wixsite.com'), ('rated 5.0', 'rated\n  great food and pleasant ambience. expensive but coll place to chill and relax......\n\nservice is really very very good and friendly staff...\n\nfood : 5/5\nservice : 5/5\nambience :5/5\noverall :5/5'), ('rated 4.0', 'rated\n  good ambience with tasty food.\ncheese chilli paratha with bhutta palak methi curry is a good combo.\nlemon chicken in the starters is a must try item.\negg fried rice was also quite tasty.\nin the mocktails, recommend ""alice in junoon"". do not miss it.'), ('rated 4.0', 'rated\n  you canã\x83ã\x83ã\x82ã\x82ã\x83ã\x82ã\x82ã\x92t go wrong with jalsa. never been a fan of their buffet and thus always order alacarteã\x83ã\x83ã\x82ã\x82ã\x83ã\x82ã\x82ã\x92. service at times can be on the slower side but food is worth the wait.'), ('rated 5.0', 'rated\n  overdelighted by the service and food provided at this place. a royal and ethnic atmosphere builds a strong essence of being in india and also the quality and taste of food is truly authentic. i would totally recommend to visit this place once.'), ('rated 4.0', 'rated\n  the place is nice and comfortable. food wise all jalea outlets maintain a good standard. the soya chaap was a standout dish. clearly one of trademark dish as per me and a must try.\n\nthe only concern is the parking. it very congested and limited to just 5cars. the basement parking is very steep and makes it cumbersome'), ('rated 4.0', 'rated\n  the place is nice and comfortable. food wise all jalea outlets maintain a good standard. the soya chaap was a standout dish. clearly one of trademark dish as per me and a must try.\n\nthe only concern is the parking. it very congested and limited to just 5cars. the basement parking is very steep and makes it cumbersome'), ('rated 4.0', 'rated\n  the place is nice and comfortable. food wise all jalea outlets maintain a good standard. the soya chaap was a standout dish. clearly one of trademark dish as per me and a must try.\n\nthe only concern is the parking. it very congested and limited to just 5cars. the basement parking is very steep and makes it cumbersome')]"
#rated 40 ratedn  a beautiful place to dine inthe interiors take you back to the mughal era the lightings are just perfectwe went there on the occasion of christmas and so they had only limited items available but the taste and service was not compromised at allthe only complaint is that the breads could have been betterwould surely like to come here again rated 40 ratedn  i was here for dinner with my family on a weekday the restaurant was completely empty ambience is good with some good old hindi music seating arrangement are good too we ordered masala papad panner and baby corn starters lemon and corrionder soup butter roti olive and chilli paratha food was fresh and good service is good too good for family hangoutncheers rated 20 ratedn  its a restaurant near to banashankari bda me along with few of my office friends visited to have buffet but unfortunately they only provide veg buffet on inquiring they said this place is mostly visited by vegetarians anyways we ordered ala carte items which took ages to come food was ok ok definitely not visiting anymore rated 40 ratedn  we went here on a weekend and one of us had the buffet while two of us took ala carte firstly the ambience and service of this place is great the buffet had a lot of items and the good was good we had a pumpkin halwa intm the dessert which was amazing must try the kulchas are great here cheers rated 50 ratedn  the best thing about the place is itãx83ãx83ãx82ãx82ãx83ãx82ãx82ãx92s ambiance second best thing was yummy  food we try buffet and buffet food was not disappointed usntest      nquality  nservice staff was very professional and friendlynnoverall experience was excellentnnsubirmajumder85wixsitecom rated 50 ratedn  great food and pleasant ambience expensive but coll place to chill and relaxnnservice is really very very good and friendly staffnnfood  55nservice  55nambience 55noverall 55 rated 40 ratedn  good ambience with tasty foodncheese chilli paratha with bhutta palak methi curry is a good combonlemon chicken in the starters is a must try itemnegg fried rice was also quite tastynin the mocktails recommend alice in junoon do not miss it rated 40 ratedn  you canãx83ãx83ãx82ãx82ãx83ãx82ãx82ãx92t go wrong with jalsa never been a fan of their buffet and thus always order alacarteãx83ãx83ãx82ãx82ãx83ãx82ãx82ãx92 service at times can be on the slower side but food is worth the wait rated 50 ratedn  overdelighted by the service and food provided at this place a royal and ethnic atmosphere builds a strong essence of being in india and also the quality and taste of food is truly authentic i would totally recommend to visit this place once rated 40 ratedn  the place is nice and comfortable food wise all jalea outlets maintain a good standard the soya chaap was a standout dish clearly one of trademark dish as per me and a must trynnthe only concern is the parking it very congested and limited to just 5cars the basement parking is very steep and makes it cumbersome rated 40 ratedn  the place is nice and comfortable food wise all jalea outlets maintain a good standard the soya chaap was a standout dish clearly one of trademark dish as per me and a must trynnthe only concern is the parking it very congested and limited to just 5cars the basement parking is very steep and makes it cumbersome rated 40 ratedn  the place is nice and comfortable food wise all jalea outlets maintain a good standard the soya chaap was a standout dish clearly one of trademark dish as per me and a must trynnthe only concern is the parking it very congested and limited to just 5cars the basement parking is very steep and makes it cumbersome
#rated 40 ratedn beautiful place dine inthe interiors take back mughal era lightings perfectwe went occasion christmas limited items available taste service compromised allthe complaint breads could betterwould surely like come rated 40 ratedn dinner family weekday restaurant completely empty ambience good good old hindi music seating arrangement good ordered masala papad panner baby corn starters lemon corrionder soup butter roti olive chilli paratha food fresh good service good good family hangoutncheers rated 20 ratedn restaurant near banashankari bda along office friends visited buffet unfortunately provide veg buffet inquiring said place mostly visited vegetarians anyways ordered ala carte items took ages come food ok ok definitely visiting anymore rated 40 ratedn went weekend one us buffet two us took ala carte firstly ambience service place great buffet lot items good good pumpkin halwa intm dessert amazing must try kulchas great cheers rated 50 ratedn best thing place itãx83ãx83ãx82ãx82ãx83ãx82ãx82ãx92s ambiance second best thing yummy food try buffet buffet food disappointed usntest nquality nservice staff professional friendlynnoverall experience excellentnnsubirmajumder85wixsitecom rated 50 ratedn great food pleasant ambience expensive coll place chill relaxnnservice really good friendly staffnnfood 55nservice 55nambience 55noverall 55 rated 40 ratedn good ambience tasty foodncheese chilli paratha bhutta palak methi curry good combonlemon chicken starters must try itemnegg fried rice also quite tastynin mocktails recommend alice junoon miss rated 40 ratedn canãx83ãx83ãx82ãx82ãx83ãx82ãx82ãx92t go wrong jalsa never fan buffet thus always order alacarteãx83ãx83ãx82ãx82ãx83ãx82ãx82ãx92 service times slower side food worth wait rated 50 ratedn overdelighted service food provided place royal ethnic atmosphere builds strong essence india also quality taste food truly authentic would totally recommend visit place rated 40 ratedn place nice comfortable food wise jalea outlets maintain good standard soya chaap standout dish clearly one trademark dish per must trynnthe concern parking congested limited 5cars basement parking steep makes cumbersome rated 40 ratedn place nice comfortable food wise jalea outlets maintain good standard soya chaap standout dish clearly one trademark dish per must trynnthe concern parking congested limited 5cars basement parking steep makes cumbersome rated 40 ratedn place nice comfortable food wise jalea outlets maintain good standard soya chaap standout dish clearly one trademark dish per must trynnthe concern parking congested limited 5cars basement parking steep makes cumbersome


'''
Now the next step is to perform some text preprocessing steps which include:

Lower casing
Removal of Punctuations
Removal of Stopwords
Removal of URLs
Spelling correction
'''

## Lower Casing
zomato["reviews_list"] = zomato["reviews_list"].str.lower()

## Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))

## Removal of Stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))


## Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))

zomato[['reviews_list', 'cuisines']].sample(5)


# RESTAURANT NAMES:
restaurant_names = list(zomato['name'].unique())
def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]
    
zomato=zomato.drop(['address','rest_type', 'type', 'menu_item', 'votes'],axis=1)
import pandas

# Randomly sample 60% of your dataframe
df_percent = zomato.sample(frac=0.5)

df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

'''
Now the last step for creating a Restaurant Recommendation System is to write a function that will recommend restaurants:
    
'''

def recommend(name, cosine_similarities = cosine_similarities):
    
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost']][df_percent.index == each].sample()))
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new
recommend('Pai Vihar')