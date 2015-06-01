import pandas as pd

# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
#%%
def hashh(text):
    return hashlib.md5(text).hexdigest()

#%%
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
import hashlib
def review_to_doc(review,  remove_stopwords=False ):
    labeledSent = LabeledSentence(words= review_to_wordlist( review, remove_stopwords ), labels=[hashh(review)])
    return labeledSent

#%%
num_features=100

#%%

from gensim.models.doc2vec import LabeledSentence
#%%
sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
print "Parsing sentences from unlabeled set"
for review in test["review"]:
    sentences.append(review_to_doc(review))
        
for review in train["review"]:
    sentences.append(review_to_doc(review))
    
print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences.append(review_to_doc(review))
    
#%%
from gensim.models import Doc2Vec
print "Training model..."
model = Doc2Vec(sentences, workers=num_workers, 
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
#%%
# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)
#%%
# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context_doc2vec"
model.save(model_name)

#%%    
#exploring the model results
model.doesnt_match("man woman child kitchen".split())
#'kitchen'    
#%%
model.doesnt_match("france london england germany".split())
#'london'
#%%
model.doesnt_match("paris berlin london austria".split())
#'paris'
#%%    
model.most_similar("man")
#%%
model.most_similar("queen")
#%%
model.most_similar("awful")

#%%
model[hashh(test["review"][0])]
#%%
def getdoc2VecFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        #
        # Print a status message every 1000th review
        if counter%1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))
        # 
        # Call the function (defined above) that makes average feature vectors
        try:
            reviewFeatureVecs[counter] = model[hashh(review)]
        except:
            print "KeyError at ."
        
        #
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs
    

#%%
train_sentences = []
for review in train["review"]:
    train_sentences.append(review)

test_sentences = []
for review in train["review"]:
    test_sentences.append(review)
#%%
model[hashh(train_sentences[43])]
#%%
trainDataVecs = getdoc2VecFeatureVecs(train_sentences, model, num_features )
#%%
testDataVecs = getdoc2VecFeatureVecs(test_sentences, model, num_features )   
#%%
trainDataVecs[3532]
#%%
testDataVecs[6556]
#%%
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 1000)
print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainDataVecs, train["sentiment"] )
# Test & extract results 
result = forest.predict( testDataVecs )
# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "docd2Vec_randomforest_1000.csv", index=False, quoting=3 )