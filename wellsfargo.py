"""
Date: 10/20/2017
Purpose: predicting an appropriate birthday gift and gift amount  based on profile and activites. Ultimate goal is get a feedback about the service 
from customer after receiving the gift. 
Author : Sheikh Rabiul Islam
Email: sislam42@students.tntech.edu
"""

import math
import csv
import pandas as pd
from textblob import TextBlob as tb

#configurations
#Assign points or weight   to different activities or profile attributes. 
#sum of points or weight need to 100
p_savings_account_balance = 30   #points given for maximum saving account balance among all customer
p_check_account_balance = 30     #points given for maximum checking account balance among all customer
p_age = 10            #points given for most senior customer   among all customer
p_tenure =15          #points given for the customer with highest tenure among all customer
p_webpage_visit = 7    #points given for maximum number of webpage visit among all customer
p_interaction = 8      #points given for maximum number of interaction with bank among all customer

#normalization ranges
to_min_point = 0    
to_max_point = 1

max_gift_value = 30     #maximum amount of gift that can be given to a customer
min_gift_value = 5      #minimum

min_point = 0           # minimum point that can be given to an activities or profile attributes
max_point = 100         #maximum

# scale a number from a range to a different range (eg. 1..100 to 0 ..1)
def scale_a_number(inpt, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(inpt-from_min)/(from_max-from_min)+to_min

# calculate term frequency (number of times as word appear in a document/record), tfidf calculation code template is taken from  http://stevenloria.com/finding-important-word-in-a-document-using-tf-idf
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)
    
# number of document/record contains the word
def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)
    
# inverse document frequency which measures how common a word is among all documents/ records
def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

#product of tf and idf which computes TF-IDF score
def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

# load credit card usage categories data
df = pd.read_csv('input/card_usage_categories.csv')

# store the data in a bloblist
bloblist = []
cr_customer_id_list = []
for index,row in df.iterrows():
    cr_customer_id_list.append(row[0])
    bloblist.append(tb(row[1]))


#load web page visit data, this is raw data which is later processed to get customer wise count/sum
df_webpage_visit = pd.read_csv('input/gift_value_by_web_page_visit_count.csv')

#load web profile data, this is raw data which is later processed to get customer wise count/sum/min/max/avg
df_profile = pd.read_csv('input/gift_value_by_profile.csv')

#load customers interaction with bank's data, this is raw data which is later processed to get customer wise count/sum/min/max/avg
df_interaction = pd.read_csv('input/gift_value_by_interaction_count.csv')


#customer wise total interaction count
interaction_gb = df_interaction.groupby('masked_id')['masked_id'].count()

#customer wise webpage visit count
webpage_visit_gb = df_webpage_visit.groupby('masked_id')['masked_id'].count()

#customerwise latest age
profile_last_age_gb = df_profile.groupby('masked_id')['age'].max()

#customerwise maximum tenure
profile_tenure_altered_gb = df_profile.groupby('masked_id')['tenure_altered'].max()

#customer wise average checking balance
profile_check_bal_altered_gb = df_profile.groupby('masked_id')['check_bal_altered'].mean()

#customer wise average savings balance
profile_sav_bal_altered_gb = df_profile.groupby('masked_id')['sav_bal_altered'].mean()


#storage for holding final result
df_result = profile_last_age_gb.to_frame()
df_result.pop("age")
df_result["calculated_gift_value"] = 0.0 #default value
df_result["gift_choice_hints"] = 'No credit card, a gift based on debit card usage categories (if any) or any gift within the calculated gift value'   #default values


#for each customer calculate most important 5 words based on credit card usage category list. 
for i, blob in enumerate(bloblist):
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    res_str = ''
    for word, score in sorted_words[:5]:
        tmp_str =  " | %s:%s" % (word,round(score,4))
        res_str += tmp_str
    res_str += ' | '    
    df_result.set_value(cr_customer_id_list[i],'gift_choice_hints',res_str)


#scale points[1..100] in to 0..1 range
n_savings_account_balance = scale_a_number(p_savings_account_balance, to_min_point, to_max_point, min_point, max_point)
n_check_account_balance = scale_a_number(p_check_account_balance, to_min_point, to_max_point, min_point, max_point)
n_age = scale_a_number(p_age, to_min_point, to_max_point, min_point, max_point)
n_tenure = scale_a_number(p_tenure, to_min_point, to_max_point, min_point, max_point)
n_webpage_visit = scale_a_number(p_webpage_visit, to_min_point, to_max_point, min_point, max_point)
n_interaction = scale_a_number(p_interaction, to_min_point, to_max_point, min_point, max_point)

#global max, min for all customers
max_check_bal_altered = df_profile.check_bal_altered.max()
min_check_bal_altered = df_profile.check_bal_altered.min()
max_sav_bal_altered = df_profile.sav_bal_altered.max()
min_sav_bal_altered = df_profile.sav_bal_altered.min()
max_tenure_altered = df_profile.tenure_altered.max()
min_tenure_altered = df_profile.tenure_altered.min()
max_age = df_profile.age.max()
min_age = df_profile.age.min()
max_webpage_visit_count = webpage_visit_gb.to_frame().masked_id.max()
min_webpage_visit_count = webpage_visit_gb.to_frame().masked_id.min()
max_interaction = interaction_gb.to_frame().masked_id.max()
min_interaction = interaction_gb.to_frame().masked_id.min()

#iterate through all the customer and calculate value of gift based on profile and activities
#for each profile attribute or activities a  sacled value [0..1] is calculated.Finally all saclled values are added [sum can be maximum 1]
for index,row in df_result.iterrows():
    masked_id = index
    inpt = 0.0
    gift_score = 0.0
    
    if masked_id  in profile_sav_bal_altered_gb.index:      #some customer has no information
        inpt =profile_sav_bal_altered_gb.at[masked_id]     
    else:
        inpt = min_sav_bal_altered
    # scale saving balance in to 0..1 range and then multiply it with eligible sacled points for this activities or profile attribute
    gift_score += n_savings_account_balance * (scale_a_number(inpt,to_min_point,to_max_point,min_sav_bal_altered,max_sav_bal_altered ))  
    
    if masked_id  in profile_check_bal_altered_gb.index:
        inpt =profile_check_bal_altered_gb.at[masked_id]
    else:
        inpt = min_check_bal_altered 
    #as before    
    gift_score += n_check_account_balance * (scale_a_number(inpt,to_min_point,to_max_point,min_check_bal_altered,max_check_bal_altered))
    
    if masked_id  in profile_last_age_gb.index:
        inpt =profile_last_age_gb.at[masked_id]
    else:
        inpt = min_age
    gift_score += n_age * (scale_a_number(inpt,to_min_point,to_max_point,min_age,max_age ))
    
    if masked_id  in profile_tenure_altered_gb.index:
        inpt =profile_tenure_altered_gb.at[masked_id]
    else:
        inpt = min_tenure_altered
    gift_score += n_tenure * (scale_a_number(inpt,to_min_point,to_max_point,min_tenure_altered,max_tenure_altered ))
    
    if masked_id  in webpage_visit_gb.index:
        inpt =webpage_visit_gb.at[masked_id]
    else:
        inpt = min_webpage_visit_count
    gift_score += n_webpage_visit * (scale_a_number(inpt,to_min_point,to_max_point,min_webpage_visit_count,max_webpage_visit_count ))
    
    if masked_id  in interaction_gb.index:
        inpt = interaction_gb.at[masked_id]
    else:
        inpt = min_interaction
    gift_score += n_interaction * (scale_a_number(inpt,to_min_point,to_max_point,min_interaction,max_interaction ))
    
    # percentage of maximum gift value a customer should get
    calc_gift_value =  gift_score *  max_gift_value  
    # a customer should net get less than minimum gift value set
    if calc_gift_value < min_gift_value:
        calc_gift_value = min_gift_value
        
    #update the calculated gift value in final result storage
    df_result.set_value(masked_id,'calculated_gift_value',round(calc_gift_value,4))

print ("please see output/result.csv file for detail result")
print(df_result)
print ("please see output/result.csv file for detail result")

#write final result in to a csv file
df_result.to_csv('output/result.csv', sep=',',encoding='utf-8')

#end of program