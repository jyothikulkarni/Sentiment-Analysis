import pandas as pd
import nltk,re,pprint,math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cosine
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

vectorizer = CountVectorizer(min_df=0, stop_words=None, strip_accents='ascii')
wordnet_lemmatizer = WordNetLemmatizer()                            #Lemmatizer Object
stop_words = set(stopwords.words('english'))                        #stop words from english
porter_stemmer = PorterStemmer()                                    #Stemming Object
df=pd.read_csv('E-Commerce_data.csv')
corpus,real,rating,product=[],[],[],[]
tk = TweetTokenizer()                                               #Tokenising Object
dictionary = {}

#building corpus, tokenising the reviews, removing symbols
for i in df.index:
    if df['Department Name'][i]=='Jackets' and str(df['Recommended IND'][i])=='1':
        review = str(df['Review Text'][i])
        real.append(review)
        rating.append(str(df['Rating'][i]))
        product.append(str(df['Clothing ID'][i]))
        review = re.sub("[^A-Za-z']+",' ',review)                   #removing all symbols
        review = ' '.join(tk.tokenize(review))                      #tokenising
        corpus.append(review) 

print('The corpus for Jackets consists of '+str(len(corpus))+' reviews.')

#remove stop words
for i in range(len(corpus)):
    querywords = corpus[i].split()
    resultwords  = [word for word in querywords if word not in stop_words]
    corpus[i] = ' '.join(resultwords)
    
#Stemming and Lemmatization
for i in range(len(corpus)):
    tokens = tk.tokenize(corpus[i])
    s=''
    for word in tokens:
        word = porter_stemmer.stem(word)
        word = wordnet_lemmatizer.lemmatize(word)
        s+=word+' '
    corpus[i]=s

docs_tf = vectorizer.fit_transform(corpus)			            #matrix form of ((document,word_index) count_of_word)
vocabulary_terms = vectorizer.get_feature_names()	            #complete vocabulary built

#Information Retrieval problem using TF-IDF matrix
print('Enter the keywords you want to search for (eg: nice big fit): ',end='')
query = input()

#Stemming and Lemmatization of the USER input
s=''
user_query = query.split()
for word in user_query:
    word = porter_stemmer.stem(word)
    word = wordnet_lemmatizer.lemmatize(word)
    s+=word+' '
query=s

# To keep the development simple, we build a composite model for both the corpus and the query 
docs_query_tf = vectorizer.transform(corpus + [query]) 
transformer = TfidfTransformer(smooth_idf = False)
tfidf = transformer.fit_transform(docs_query_tf.toarray())

tfidf_matrix = tfidf.toarray()[:-1]                             # D x V document-term TF-IDF matrix 
query_tfidf = tfidf.toarray()[-1]                               # 1 x V query-term vector 

#getting the cosine values between the query and tf-idf vectors.. then ranking them based on these values
query_doc_tfidf_cos_dist = [cosine(query_tfidf, doc_tfidf) for doc_tfidf in tfidf_matrix]
query_doc_tfidf_sort_index = np.argsort(np.array(query_doc_tfidf_cos_dist))

dist1=[]
for rank, sort_index in enumerate(query_doc_tfidf_sort_index):
    dist1.append(query_doc_tfidf_cos_dist[sort_index])
    
#Information Retrieval problem using LSA-Kmeans method
tf_matrix = docs_tf.toarray()                                   #D x V term frequency matrix 
query_vector = tf_matrix.T                                                 #LSA using SVD on Term Frequency (TF) matrix

U, s, V = np.linalg.svd(query_vector, full_matrices=1, compute_uv=1)	    # U, sigma, V

K = 400  #reduce dimensions

query_vector_reduced = np.dot(U[:,:K], np.dot(np.diag(s[:K]), V[:K, :]))   # D x V matrix 
docs_rep = np.dot(np.diag(s[:K]), V[:K, :]).T                   # D x K matrix 
terms_rep = np.dot(U[:,:K], np.diag(s[:K]))                     # V x K matrix 

key_words = query.split()
key_word_indices = [vocabulary_terms.index(key_word) for key_word in key_words if key_word in vocabulary_terms] # vocabulary indices 
key_words_rep = terms_rep[key_word_indices,:]                   #getting the vocabulary values on these indices
query_rep = np.sum(key_words_rep, axis = 0)

# K means
query_vector=[]
query_vector.append(list(query_rep))
query_vector.append([0]*K)

X = docs_rep
kmeans = KMeans(n_clusters=10, n_init=1, init=X[:10]).fit(X)
means = kmeans.cluster_centers_
label = kmeans.labels_
x=kmeans.predict(query_vector)
ans=[]
for i in range(len(label)):
    if label[i] == x[0]:
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(tuple(query_rep), tuple(X[i]))]))
        ans.append((i,distance))

ans = (sorted(ans, key=lambda x: (x[1],x[1])))
print(len(ans))
k=4
if(len(ans)<k):
	k=len(ans)
products=[]
c=0
print('\nTOP 4 Recommendations are as follows:')
# Products to be recommended... Clothing ID, Review, Rating
while(c<k):
	if product[ans[c][0]] in products:
		continue
	products.append(product[ans[c][0]])
	print('Rating : '+str(rating[ans[c][0]]))
	print('Product : '+str(product[ans[c][0]]))
	print('Review : '+str(real[ans[c][0]]))
	print('\n')
	c+=1

kmeans_rep = []
#getting the cosine values between the query and tf-idf vectors.. then ranking them based on these values
query_doc_cos_dist = [cosine(query_rep, doc_rep) for doc_rep in docs_rep]
query_doc_sort_index = np.argsort(np.array(query_doc_cos_dist))

for i in ans:
	kmeans_rep.append((query_doc_cos_dist[i[0]],i[0]))

dist3=[]
for j in kmeans_rep:
	dist3.append(query_doc_tfidf_cos_dist[j[1]])

#LSA method
dist2=[]
for rank, sort_index in enumerate(query_doc_sort_index):
    dist2.append(query_doc_tfidf_cos_dist[sort_index])

#Plot to compare TF-IDF and LSA
fig = plt.figure(1)
plt.title('TF-IDF v/s LSA v/s kmeans')
plt.plot(dist1,'ro',label='tfidf')
plt.plot(dist3,'go',label='kmeans')
plt.plot(dist2,'bo',label='lsa')
plt.xlabel("Rank")
plt.ylabel("Distance (1-cosθ)")
plt.legend()
plt.show()
