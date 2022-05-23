# Loading Libraries
import pandas as pd
import numpy as np
import re
import gensim 
from gensim.parsing.preprocessing import remove_stopwords 
import gensim.downloader as api
import sklearn
from sklearn.metrics.pairwise import cosine_similarity;

# Loading the Q&A data set
df = pd.read_csv("Data/DS_interview.csv");

# Renaming columns of data set
df.columns = ["Q.No", "questions","answers"];

# Question & answer data set
print("--------------------------------------------Question & Answer pairs------------------------------------------------\n")
for i in range(1,11):
    Ques = df.loc[df["Q.No"]== i]
    print('\033[1m' + "Question",i,":",Ques["questions"].iloc[0] + "\033[0;0m")
    print('\033[1m' + "\nStandard Answer:",Ques["answers"].iloc[0] + "\033[0;0m")
    print("\n\n")
    
#Function to choose question-answer pair from given data set
def Q_selection(Q):
    # Loop to change input, if not in given range
    while(True):
        Question_Number = input("Enter the question number you want to ask from above the list of questions:")
        Q = int(Question_Number)
        
        # Error message
        if Q > 10 or Q < 1:
            print('\033[1m' + "Error: Please enter a number from 1 to 10\n"+ "\033[0;0m")
        else:
            break
    
    print("\n---------------------You have selected following Q&A pair:-----------------\n")
        
    Ques = df.loc[df["Q.No"]== Q]
    print('\033[1m' + "Question:",Ques["questions"].iloc[0] + "\033[0;0m")
    print('\033[1m' + "\nStandard Answer:",Ques["answers"].iloc[0] + "\033[0;0m")
        
    return Q    # Return the choosen question number

# Choosing question-answer pair number
N = Q_selection(-1)

# Loading the sample answers data set for the choosen Q&A pair, N
sample = pd.read_csv(('Data/Sample_'+str(N)+'.csv'));

# Renaming columns of sample answers data set
sample.columns=["marks","answers"];

# Cleaning single sentence
def clean_sentence(sentence, stopwords = False):
    # Converting all characters to lower case
    sentence = sentence.lower().strip()
    
    # Replace non-alpha numeric characters with spaces
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    
    # Removing stopwords if asked to do so
    if stopwords:
         sentence = remove_stopwords(sentence)
    
    # Returning cleaned sentence
    return sentence
            
# Cleaning data frame of sentences
def get_cleaned_sentences(df,stopwords = False):
    # Choosing answers column from data frame
    sents = df[["answers"]];
    cleaned_sentences=[]   # Initializing list to store cleaned sentences

    # Looping over rows of data frame
    for index,row in df.iterrows():
        # Cleaning each sentence
        cleaned = clean_sentence(row["answers"],stopwords);
        
        # Appending cleaned sentence to the cleaned_sentences list
        cleaned_sentences.append(cleaned);
    
    # Returning data frame of cleaned sentences
    return cleaned_sentences;

# Cleaning all sample answers of the choosen Q&A pair
cleaned_sample_ans = get_cleaned_sentences(sample, stopwords = True)

# Function to take students answer of asked question as input
def student_answer():
    student_ans = input("---------Student should type his answer below and press enter---------\n")
    
    # Cleaning students answer
    student_ans = clean_sentence(student_ans, stopwords = True);
    
    # Returning cleaned student answer
    return student_ans

# Reading & cleaning students answer
cleaned_student_ans = student_answer()

# Making glove embedding model
glove_model = None;   # Initialization

# Checking if model is already downloaded
try:
    glove_model = gensim.models.KeyedVectors.load("./glovemodel.mod")
    print("Loaded glove model")
    
# Download the model if not already downloaded
except:            
    glove_model = api.load('glove-twitter-25')
    glove_model.save("./glovemodel.mod")
    print("Saved glove model")
    
# Function to convert word to vector by given model
def getWordVec(word, model):
    # Reading downloaded model from computer
    samp = model['computer'];
    
    # Initializing vector
    vec = [0]*len(samp);
    
    # Checking if word is already present in model
    try:
            vec = model[word];
    except:
            vec = [0]*len(samp);
    
    # Returning the word vector
    return vec

# Function to convert word embeddings to phrase embeddings
def getPhraseEmbedding(phrase, embeddingmodel):

        samp = getWordVec('computer', embeddingmodel);
        vec = np.array([0]*len(samp));
        den = 0;
        
        # Looping over all words of given phrase
        for word in phrase.split():
            den = den + 1;
            vec = vec + np.array(getWordVec(word,embeddingmodel));
            
        # Returning the phrase embedding
        return vec.reshape(1, -1)
    
# Function to displaying cosine similaritiy & final marks obtained
def retrieveAndPrintmarks(student_embedding, sample_embedding, sample_df):
    print("Cosine similarity between students answer & each sample answer for Q"+str(N)+" are as follows:\n")
    
    max_sim = -1;   # Initializing maximum value of cosine similarity
    index_sim = -1;  # Initializing index corresponding to maximum cosine similarity
    
    # Looping over all sample embeddings
    for index, samp_embedding in enumerate(sample_embedding):
        
        # Computing cosine similarity between student answer & sample answer
        sim = cosine_similarity(samp_embedding, student_embedding)[0][0];
        
        # Displaying cosine similarity for each sample answer
        print('\033[1m' + "Sample answer", index + 1, "(" + str(10 - index*2) + "/10):", sim,"\033[0;0m")
        
        # Changing the maximum value of cosine similarity & corresponding index in sample_df
        if sim > max_sim:
            max_sim = sim;
            index_sim = index;
    
    # Computing & displaying student mark
    value = sample_df.iloc[index_sim,0] * max_sim
    print('\033[1m', "\nStudent Marks out of 10: ","{:.2f}".format(value),"\033[0;0m")
    
    # Displying formula to compute student mark
    print("\nNote: Formula used for marks calculation is as follows")
    print("\tMarks = (Marks of retrieved sample answer)*(Corresponding cosine similairty)")
    
# Applying Glove embedding model
sample_embeddings = [];   # Initializing list of sample answer embeddings

# Looping over sample answers
for sent in cleaned_sample_ans:
    # Computing & appending each sample answer embedding to sample_embeddings using Glove model
    sample_embeddings.append(getPhraseEmbedding(sent, glove_model));

# Computing student answer embedding using Glove model
student_embedding = getPhraseEmbedding(cleaned_student_ans, glove_model);

# Displaying final results
retrieveAndPrintmarks(student_embedding, sample_embeddings, sample);