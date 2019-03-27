import pandas as pd
data=pd.read_csv('/Users/rain/Desktop/shooting.csv')
gun=pd.read_csv('/Users/rain/Desktop/gun.csv')


# NLP find the keywords in the shootings
summary=data['Summary']
list=[]
reviews=''
# clean the text
for i in range(len(summary)):
    import re
    import nltk
    nltk.download('stopwords')  # list of words we donot need in our review
    from nltk.corpus import stopwords
    review = re.sub('[^a-zA-Z]', ' ', summary[i])
    # put letter in lower case
    review = review.lower()
    # remove word we donot need
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)  # turn the list into str
    list.append(review)
    reviews=reviews+review

# word cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import codecs

cloud = WordCloud(
        background_color='white',
        max_words=2000,
        max_font_size=80,
    )

reviews = reviews.split()
cut_text = " ".join(nltk.Text(reviews))
word_cloud = cloud.generate(cut_text) #
plt.imshow(word_cloud)
plt.axis('off')
plt.title('keywords in shooting')
plt.show()

# NLP find the keywords in the mental health
stopword=pd.read_fwf('/Users/rain/Desktop/stop.txt')
mental=data['Mental Health']
list=[]
reviews=''
# clean the text
for i in range(len(mental)):
    if not pd.isnull(mental[i]):
        import re
        import nltk
        nltk.download('stopwords')  # list of words we donot need in our review
        from nltk.corpus import stopwords

        review = re.sub('[^a-zA-Z]', ' ', mental[i])  
        # put letter in lower case
        review = review.lower()
        # remove word we donot need
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopword)]
        review = ' '.join(review)  # turn the list into str
        list.append(review)
        reviews = reviews + review

# word cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import codecs

cloud = WordCloud(
        background_color='white',
        max_words=2000,
        max_font_size=80,
    )

reviews = reviews.split()
cut_text = " ".join(nltk.Text(reviews))
word_cloud = cloud.generate(cut_text) # 产生词云
plt.imshow(word_cloud)
plt.axis('off')
plt.title('keywords in mental health')
plt.show()


# compared gun data with shooting
year=gun['year'].unique()
year_all=pd.DataFrame(gun[gun['year']==2000].mean(axis=0))
for ele in year[1:]:
    df=pd.DataFrame(gun[gun['year']==ele].mean(axis=0))
    year_all=pd.concat([year_all,df],axis=1)

result=year_all.transpose()
result=result.drop(['month'],axis=1)
result.to_csv('gun_year.csv')



# after cleaning data:( drop some cols)
new=data.loc[:,['Case', 'Location', 'Year', 'Summary', 'Fatalities', 'Injured', 'Total victims', 'Venue',
                'Prior signs of possible mental illness', 'Mental Health', 'Weapons obtained legally', 'Type of weapons', 'Race', 'Gender']]


# ﻿one dimension plot
#  boxplot  of victims
plt.figure()
plt.subplot(1,3,1)
plt.boxplot(new['Total victims'])
plt.title('Total victims')
plt.subplot(1,3,2)
plt.boxplot(new['Fatalities'])
plt.title('Fatalities')
plt.subplot(1,3,3)
plt.boxplot(new['Injured'])
plt.title('Injured')
plt.tight_layout()
plt.show()


# venue using pie
labels = ['School', 'Military', 'Religious', 'Other', 'Workplace']
len(new[new['Venue']=='School'])
len(new[new['Venue']=='Military'])
len(new[new['Venue']=='Religious'])
len(new[new['Venue']=='Other'])
len(new[new['Venue']=='Workplace'])
sizes = [14, 4, 4, 30,20]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0,0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.title('venue')
plt.show()


# race pie plot
# change nan to unknown
for i in range(len(new['Race'])):
    if pd.isnull(new['Race'][i]):
        new['Race'][i] = 'Unknown'


labels = ['Unknown', 'Other', 'white', 'Native American', 'black', 'Latino', 'Asian']
len(new[new['Race']=='Unknown'])
len(new[new['Race']=='Other'])
len(new[new['Race']=='white'])
len(new[new['Race']=='Native American'])
len(new[new['Race']=='black'])
len(new[new['Race']=='Latino'])
len(new[new['Race']=='Asian'])

sizes = [2, 2, 44, 3,11,4,6]
explode = (0.1, 0, 0, 0,0,0,0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.title('Race')
plt.show()



# gender pie plot
labels = ['Male', 'Female']
len(new[new['Gender']=='Female'])


sizes = [70,2]
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.title('Gender')
plt.show()


# analysis which character will affect the victims

# anova table for  venue
from scipy import stats
school=(new[new['Venue']=='School'])['Total victims']
Military=(new[new['Venue']=='Military'])['Total victims']
Religious=(new[new['Venue']=='Religious'])['Total victims']
Other=(new[new['Venue']=='Other'])['Total victims']
Workplace=(new[new['Venue']=='Workplace'])['Total victims']
argss=(school, Military, Religious, Other, Workplace)
# var
stats.levene(*argss)
#anova table
stats.f_oneway(*argss)    # 地点对于victims有影响

# contrast ( compare school and other locations)
argss=(school, Religious)
stats.levene(*argss)
stats.f_oneway(*argss)

argss=(school, Other)
stats.levene(*argss)
stats.f_oneway(*argss)

argss=(school, Workplace)
stats.levene(*argss)
stats.f_oneway(*argss)





plt.figure()
plt.subplot(1,5,1)
plt.boxplot(school)
plt.title('school')
plt.subplot(1,5,2)
plt.boxplot(Military)
plt.title('Military')
plt.subplot(1,5,3)
plt.boxplot(Religious)
plt.title('Religious')
plt.subplot(1,5,4)
plt.boxplot(Other)
plt.title('Other')
plt.subplot(1,5,5)
plt.boxplot(Workplace)
plt.title('Workplace')
plt.tight_layout()
plt.show()


# contrast


# anova table for race
unknown=(new[new['Race']=='Unknown'])['Total victims']
other=(new[new['Race']=='Other'])['Total victims']
white=(new[new['Race']=='white'])['Total victims']
native=(new[new['Race']=='Native American'])['Total victims']
black=(new[new['Race']=='black'])['Total victims']
Latino=(new[new['Race']=='Latino'])['Total victims']
Asian=(new[new['Race']=='Asian'])['Total victims']
argss=(unknown, other, white, native, black, Latino, Asian)
# var
stats.levene(*argss)
#anova table
stats.f_oneway(*argss)    # race not affect


# anova table for gender
male=new[new['Gender']=='Male']['Total victims']
female=new[new['Gender']=='Female']['Total victims']
argss=(male,female)
# var
stats.levene(*argss)
#anova table
stats.f_oneway(*argss)    # gender not affect


# MLR
y=new['Total victims'].values
X=new.iloc[:,[7,8,10,12,13]].values
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X = labelencoder_X.fit_transform(X)
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3],sparse=False)
X = onehotencoder.fit_transform(X)

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    '''x is the predictors
    return x '''
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

# model
x=backwardElimination(X, 0.05)
regressor_OLS = sm.OLS(y, x).fit()
regressor_OLS.summary()











