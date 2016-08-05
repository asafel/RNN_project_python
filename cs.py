from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# original_ejohn_lyrics

original_ejohn_lyrics_sentences = open('original_ejohn_lyrics.txt','r').read().split('\n')
print(original_ejohn_lyrics_sentences)

# model_ejohn_lyrics

model_ejohn_lyrics_sentences = open('model_elton.txt','r').read().split('\n')
#model_ejohn_lyrics_sentences = open('model_ejohn_lyrics.txt','r').read().split('\n')
print(model_ejohn_lyrics_sentences)


v = TfidfVectorizer()
v.fit(original_ejohn_lyrics_sentences)

Y = v.transform(original_ejohn_lyrics_sentences)

sum = 0
for sentence in model_ejohn_lyrics_sentences:
    X = v.transform([sentence])
    print(sentence)
    print(original_ejohn_lyrics_sentences[cosine_similarity(X, Y).argmax()])
    print(' ')
    sum = cosine_similarity(X, Y).max()


similarity = sum/500
print('The similarity between the original and model is ' , similarity)

#X = v.transform(["english, do you speak it?"])
#W = ["do you speak english, motherfucker?", "hello world", "english world"]
#Y = v.transform(W)

#Y = v.transform(["do you speak english, motherfucker?", "hello world", "english world"])
#print(cosine_similarity(X, Y))
#print(cosine_similarity(X, Y).argmax())
#print(W[cosine_similarity(X, Y).argmax()])

# sum of all cosine and divide to (number of sencence which where compare)