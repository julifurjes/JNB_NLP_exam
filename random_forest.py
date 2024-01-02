import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel, ttest_ind
import os
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from wordcloud import WordCloud
import string

# Print working directory
print("Current working directory:", os.getcwd())

# Import data
data = pd.read_excel('job_ads_updated.xlsx')
print(len(data))

##### PREPROCESSING #####

# Remove rows with NaN in 'competencies' and 'title'
data.dropna(subset=['competencies'], inplace=True)
data.dropna(subset=['title'], inplace=True)

# Calculate the ratio of male to female applicants for each job post
data['gender_ratio'] = data['females'] / (data['females'] + data['males'])
# Remove NaNs
data.dropna(subset=['gender_ratio'], inplace=True)

# Separate the data into 'competencies' and 'title' datasets
data_comp = data[['competencies', 'gender_ratio']]
data_title = data[['title', 'gender_ratio']]

# Load in feminine and masculine prefixes dataset
with open('feminine_words.txt', 'r') as feminine_file:
    # Read the lines from the file and remove '-' at the end of the words
    feminine_word_list = [line.strip().rstrip('-') for line in feminine_file.readlines()]

with open('masculine_words.txt', 'r') as masculine_file:
    # Read the lines from the file and remove '-' at the end of the words
    masculine_word_list = [line.strip().rstrip('-') for line in masculine_file.readlines()]

##### DATA ANALYSIS #####

# Count how many women and men applicants are there altogether

all_women = data['females'].sum()
all_men = data['males'].sum()

print('Total women applicants: ', all_women)
print('Total men applicants: ', all_men)

# Count the unique occurrences of the job family
family_counts = data['family'].value_counts()

print(family_counts)

# Create a pie chart for job family

# Define the threshold as less than 5% for 'Other' category
threshold = family_counts.sum() * 0.05

# Create a new Series for the pie chart with an 'Other' category
family_counts_adj = family_counts.copy()
family_counts_adj['Other'] = 0
categories_to_remove = []
for category in family_counts.index:
    if family_counts[category] < threshold:
        family_counts_adj['Other'] += family_counts[category]
        categories_to_remove.append(category)
family_counts_adj = family_counts_adj.drop(categories_to_remove)

colors = plt.cm.Paired(range(len(family_counts_adj)))  # Colour palette
plt.figure(figsize=(20, 15))  # Figure size
wedges, texts, autotexts = plt.pie(family_counts_adj, labels=family_counts_adj.index, colors=colors, autopct='%1.1f%%', startangle=140)

# Increase font sizes for labels and percentages in the pie chart
plt.setp(texts, fontsize=20)  # Increase font size for labels
plt.setp(autotexts, fontsize=20)  # Increase font size for percentages

# Make a legend with job family names for small slices included in 'Other'
handles = [Patch(facecolor=col, label=f'{family}: {count} ({(count/family_counts_adj.sum())*100:1.1f}%)') 
           for family, (col, count) in zip(family_counts_adj.index, zip(colors, family_counts_adj))]
plt.legend(handles=handles, title="Job Families", loc="center left", bbox_to_anchor=(1.05, 1), fontsize=20, title_fontsize=16)

plt.axis('equal') 
plt.title('Distribution of Job Families', fontsize=30)

# Adjust the layout
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)

# Apply tight layout
plt.tight_layout()

# Save plot
plt.savefig('job_family_pie_chart.png')

##### SET UP MODEL #####

# Initialize and train the Random Forest model for 'competencies'
X_comp = data_comp['competencies']
y_comp = data_comp['gender_ratio']
X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(X_comp, y_comp, test_size=0.1, random_state=42)

vectorizer_comp = CountVectorizer()
X_train_comp_encoded = vectorizer_comp.fit_transform(X_train_comp).toarray()
X_test_comp_encoded = vectorizer_comp.transform(X_test_comp).toarray()

rf_model_comp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_comp.fit(X_train_comp_encoded, y_train_comp)

# Initialize and train the Random Forest model for 'title'
X_title = data_title['title']
y_title = data_title['gender_ratio']
X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(X_title, y_title, test_size=0.1, random_state=42)

vectorizer_title = CountVectorizer()
X_train_title_encoded = vectorizer_title.fit_transform(X_train_title).toarray()
X_test_title_encoded = vectorizer_title.transform(X_test_title).toarray()

rf_model_title = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_title.fit(X_train_title_encoded, y_train_title)

##### PREDICTION #####

# Make predictions on the training set for 'competencies'
y_train_pred_comp = rf_model_comp.predict(X_train_comp_encoded)

# Make predictions on the test set for 'competencies'
y_test_pred_comp = rf_model_comp.predict(X_test_comp_encoded)

# Make predictions on the training set for 'title'
y_train_pred_title = rf_model_title.predict(X_train_title_encoded)

# Make predictions on the test set for 'title'
y_test_pred_title = rf_model_title.predict(X_test_title_encoded)

# For comparing title and description
y_pred_comp = rf_model_comp.predict(X_test_comp_encoded)
y_pred_title = rf_model_title.predict(X_test_title_encoded)

print('Predicted competencies (mean): ', sum(y_pred_comp)/len(y_pred_comp))
print('Predicted titles (mean): ', sum(y_pred_title)/len(y_pred_title))

t_stat, p_value = ttest_ind(y_test_pred_comp, y_test_pred_title)

print(f"T-statistic for comparison between job descriptions and job titles: {t_stat}")
print(f"P-value for this comparison: {p_value}")

# Interpretation of the result
alpha = 0.05  # significance level
if p_value < alpha:
    print("There is a statistically significant difference in predicting gender ratio between job descriptions and job titles.")
else:
    print("There is no statistically significant difference in predicting gender ratio between job descriptions and job titles.")

##### EVALUATION #####

# Evaluate the model for 'competencies'
mse_train_comp = mean_squared_error(y_train_comp, y_train_pred_comp)
mse_test_comp = mean_squared_error(y_test_comp, y_test_pred_comp)

print(f"Training MSE for 'competencies': {mse_train_comp}")
print(f"Testing MSE for 'competencies': {mse_test_comp}")

# Ensure y_train_pred and y_test_pred have the same length
min_length = min(len(y_train_pred_comp), len(y_test_pred_comp))
y_train_pred_comp = y_train_pred_comp[:min_length]
y_test_pred_comp = y_test_pred_comp[:min_length]

# Perform t-test for 'competencies'
t_stat_comp, p_value_comp = ttest_rel(y_train_pred_comp, y_test_pred_comp)

print(f"T-statistic for 'competencies': {t_stat_comp}")
print(f"P-value for 'competencies': {p_value_comp}")

# Check for significance for 'competencies'
alpha = 0.05  # Set your significance level
if p_value_comp < alpha:
    print("The difference between MSE_train and MSE_test for 'competencies' is statistically significant.")
else:
    print("There is no statistically significant difference between MSE_train and MSE_test for 'competencies'.")

# Cross-validate for 'competencies'
encoder = OneHotEncoder()

comp_encoded = encoder.fit_transform(data[['competencies']]).toarray()

scores_comp = cross_val_score(rf_model_comp, comp_encoded, data['gender_ratio'], cv=5, scoring='neg_mean_squared_error')
rmse_scores_comp = np.sqrt(-scores_comp)
print("Cross-validated RMSE for 'competencies':", rmse_scores_comp)
print("Mean RMSE for 'competencies':", rmse_scores_comp.mean())

# Evaluate the model for 'title'
mse_train_title = mean_squared_error(y_train_title, y_train_pred_title)
mse_test_title = mean_squared_error(y_test_title, y_test_pred_title)

print(f"Training MSE for 'title': {mse_train_title}")
print(f"Testing MSE for 'title': {mse_test_title}")

# Ensure y_train_pred and y_test_pred have the same length
min_length = min(len(y_train_pred_title), len(y_test_pred_title))
y_train_pred_title = y_train_pred_title[:min_length]
y_test_pred_title = y_test_pred_title[:min_length]

# Perform t-test for 'title'
t_stat_title, p_value_title = ttest_rel(y_train_pred_title, y_test_pred_title)

print(f"T-statistic for 'title': {t_stat_title}")
print(f"P-value for 'title': {p_value_title}")

# Check for significance for 'title'
if p_value_title < alpha:
    print("The difference between MSE_train and MSE_test for 'title' is statistically significant.")
else:
    print("There is no statistically significant difference between MSE_train and MSE_test for 'title'.")

# Cross-validate for 'title'
title_encoded = encoder.fit_transform(data[['title']]).toarray()

scores_title = cross_val_score(rf_model_title, title_encoded, data['gender_ratio'], cv=5, scoring='neg_mean_squared_error')
rmse_scores_title = np.sqrt(-scores_title)
print("Cross-validated RMSE for 'title':", rmse_scores_title)
print("Mean RMSE for 'title':", rmse_scores_title.mean())

##### Gender Ratio Prediction Hypothesis #####
mse_comp = mean_squared_error(y_test_comp, y_pred_comp)
mse_title = mean_squared_error(y_test_title, y_pred_title)

print(f"MSE for Job Description Model: {mse_comp}")
print(f"MSE for Job Title Model: {mse_title}")

# Calculate differences between MSEs
mse_diff = mse_comp - mse_title

# Perform a paired t-test
t_stat, p_value = ttest_rel(y_test_comp - y_pred_comp, y_test_title - y_pred_title)

print(f"Mean Squared Error Difference: {mse_diff}")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

boxplot_data = [y_pred_comp, y_pred_title]
plt.boxplot(boxplot_data, labels=['Competencies', 'Titles'])
plt.title('Predicted Gender Ratios for Competencies and Titles')
plt.ylabel('Gender Ratio')
plt.savefig('pred_gender_ratio.png')

##### Frequency of Gendered Words Hypothesis #####

# Part one

# Define a threshold for gender ratio to classify job descriptions as biased towards male
gender_ratio_threshold = 0.3  # This actually means 70% males, as the original percentage is for the female applicants

# Create a DataFrame for job descriptions with gender ratio below the threshold (biased towards male)
biased_male_desc = data[data['gender_ratio'] <= gender_ratio_threshold]['competencies']

# Combine the job descriptions into a single string
biased_male_text = ' '.join(biased_male_desc)

# Tokenize and count words for job descriptions biased towards male
stop_words = set(stopwords.words('english'))
vectorizer = CountVectorizer(stop_words=list(stop_words))  # Convert the set to a list

# Fit and transform the vectorizer on job descriptions biased towards male
male_word_counts = vectorizer.fit_transform([biased_male_text])

# Get the vocabulary (words)
vocab = vectorizer.get_feature_names_out()

# Calculate the word frequencies
word_freq_male = male_word_counts.toarray()[0]

# Create a DataFrame to store words and their frequencies
word_freq_df = pd.DataFrame({'Word': vocab, 'Frequency': word_freq_male})

# Sort by frequency to find the most biased words
most_biased_words_male = word_freq_df.sort_values(by='Frequency', ascending=False)

# Print the top N most biased words biased towards male (after removing stopwords)
top_n = 10  # You can adjust this value to see more or fewer words
print(f"Top {top_n} Words Biased Towards Male in 'competencies' (after removing stopwords):")
print(most_biased_words_male.head(top_n))

# Part two

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return tokens

# Function to count prefix frequencies
def count_prefix_frequencies(data, prefix_list):
    prefix_freq = {prefix: 0 for prefix in prefix_list}
    for description in data['competencies']:
        tokens = preprocess_text(description)
        for word in tokens:
            for prefix in prefix_list:
                if word.startswith(prefix):
                    prefix_freq[prefix] += 1
    total_frequency = sum(prefix_freq.values())
    return prefix_freq, total_frequency

# Filter the data to include only rows with 70% or more male 'gender_ratio'
male_dominant_data = data[data['gender_ratio'] >= 0.3]

# Count frequencies for masculine words in male_dominant_data
masculine_prefix_freq_male_dominant, total_male_freq_male_dominant = count_prefix_frequencies(male_dominant_data, masculine_word_list)
print("Masculine Prefix Frequencies in Male-Dominant Data:", masculine_prefix_freq_male_dominant)

# Count frequencies for feminine words in male_dominant_data
feminine_prefix_freq_male_dominant, total_female_freq_male_dominant = count_prefix_frequencies(male_dominant_data, feminine_word_list)
print("Feminine Prefix Frequencies in Male-Dominant Data:", feminine_prefix_freq_male_dominant)

# Count frequencies for masculine words in entire data
masculine_prefix_freq_data, total_male_freq_data = count_prefix_frequencies(data, masculine_word_list)
print("Masculine Prefix Frequencies in Entire Data:", masculine_prefix_freq_data)

# Count frequencies for feminine words in entire data
feminine_prefix_freq_data, total_female_freq_data = count_prefix_frequencies(data, feminine_word_list)
print("Feminine Prefix Frequencies in Entire Data:", feminine_prefix_freq_data)

print("Total Masculine-associated Word Frequency in Male-Dominant Data:", total_male_freq_male_dominant)
print("Total Feminine-associated Word Frequency in Male-Dominant Data:", total_female_freq_male_dominant)

print("Total Masculine-associated Word Frequency in Entire Data:", total_male_freq_data)
print("Total Feminine-associated Word Frequency in Entire Data:", total_female_freq_data)

# Plotting results
# Frequencies from your results
total_freq_original = [total_male_freq_data, total_female_freq_data]  # Total frequencies in the original dataset (Male, Female)
total_freq_male_dominant = [total_male_freq_male_dominant, total_female_freq_male_dominant]  # Frequencies in the subset with mostly male applicants (Male, Female)

# Labels for your bars
labels = ['Original Dataset', 'Male-Dominant Dataset']

x = [0, 1]  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))

# Custom colors
male_color = 'lightblue'  # Light blue color for Male-Associated Prefixes
female_color = 'pink'     # Pink color for Female-Associated Prefixes

# Plotting for the Original Dataset (Male and Female)
rects1 = ax.bar(x[0] - width/2, total_freq_original[0], width, color=male_color)
rects2 = ax.bar(x[0] + width/2, total_freq_original[1], width, color=female_color)

# Plotting for the Male-Dominant Dataset (Male and Female)
rects3 = ax.bar(x[1] - width/2, total_freq_male_dominant[0], width, color=male_color)
rects4 = ax.bar(x[1] + width/2, total_freq_male_dominant[1], width, color=female_color)

# Add some text for labels, title, and custom x-axis tick labels
ax.set_ylabel('Frequencies')
ax.set_title('Frequency of Gender-Associated Prefixes by Dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(['Male-Associated Prefixes', 'Female-Associated Prefixes'])

# Function to automatically attach a label above each bar
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Attach labels
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# Save the plot
plt.savefig('frequency_of_gendered_prefixes.png')

# Wordcloud

# Convert the most_biased_male DataFrame to a dictionary
word_freq_dict = most_biased_words_male.set_index('Word')['Frequency'].to_dict()

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate_from_frequencies(word_freq_dict)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

plt.savefig('wordcloud_men.png')

##### VISUALISATIONS #####

# For description

# Create dataframe
ratios_for_plot = pd.DataFrame({'Actual Ratio': y_test_comp, 'Predicted Ratio': y_test_pred_comp})
# Sort by actual gender ratio
ratios_for_plot.sort_values(by='Actual Ratio', inplace=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(len(ratios_for_plot)), ratios_for_plot['Actual Ratio'], label='Real Ratio', marker='o', markersize=5, linestyle='-')
plt.plot(range(len(ratios_for_plot)), ratios_for_plot['Predicted Ratio'], label='Predicted Ratio', marker='o', markersize=5, linestyle='-')
plt.xlabel('Job Post Index (Sorted by Real Ratio)')
plt.ylabel('Gender Ratio')
plt.title('Actual vs. Predicted Gender Ratio for Descriptions (Sorted by Real Ratio)')
plt.legend()
plt.grid(True)
plt.show()

# Save the plot as a PNG file
plt.savefig('gender_ratio_comparison_desc.png')

std_dev = np.std(ratios_for_plot['Actual Ratio'] - ratios_for_plot['Predicted Ratio'])
print('Standard deviation between predicted and actual ratio (description): ', std_dev)

# For title

# Create dataframe
ratios_for_plot = pd.DataFrame({'Actual Ratio': y_test_title, 'Predicted Ratio': y_test_pred_title})
# Sort by actual gender ratio
ratios_for_plot.sort_values(by='Actual Ratio', inplace=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(len(ratios_for_plot)), ratios_for_plot['Actual Ratio'], label='Real Ratio', marker='o', markersize=5, linestyle='-')
plt.plot(range(len(ratios_for_plot)), ratios_for_plot['Predicted Ratio'], label='Predicted Ratio', marker='o', markersize=5, linestyle='-')
plt.xlabel('Job Post Index (Sorted by Real Ratio)')
plt.ylabel('Gender Ratio')
plt.title('Actual vs. Predicted Gender Ratio for Titles (Sorted by Real Ratio)')
plt.legend()
plt.grid(True)
plt.show()

# Save the plot as a PNG file
plt.savefig('gender_ratio_comparison_title.png')

std_dev = np.std(ratios_for_plot['Actual Ratio'] - ratios_for_plot['Predicted Ratio'])
print('Standard deviation between predicted and actual ratio (title): ', std_dev)

##### SANITY CHECK #####

# Female
feminine_predicted_ratios = 0
len = 0

for word in feminine_word_list:
    # Create a job description containing only the current word
    job_description = [word]
    
    # Transform the job description using the same vectorizer
    job_description_encoded = vectorizer_comp.transform(job_description).toarray()
    
    # Predict the gender ratio for the current word
    feminine_predicted_ratios = rf_model_comp.predict(job_description_encoded)[0]
    
    feminine_predicted_ratios += feminine_predicted_ratios
    len += 1
    
# Calculate and print the mean of the predicted gender ratios
mean_feminine_predicted_ratios = feminine_predicted_ratios/len
print(f"Mean Predicted Gender Ratio for the Feminine Words: {mean_feminine_predicted_ratios}")

# Male
masculine_predicted_ratios = 0
len = 0

for word in masculine_word_list:
    # Create a job description containing only the current word
    job_description = [word]
    
    # Transform the job description using the same vectorizer
    job_description_encoded = vectorizer_comp.transform(job_description).toarray()
    
    # Predict the gender ratio for the current word
    masculine_predicted_ratio = rf_model_comp.predict(job_description_encoded)[0]
    
    masculine_predicted_ratios += masculine_predicted_ratio
    len += 1
    
# Calculate and print the mean of the predicted gender ratios
mean_masculine_predicted_ratios = masculine_predicted_ratios/len
print(f"Mean Predicted Gender Ratio for the Masculine Words: {mean_masculine_predicted_ratios}")