
# Final Project

Asaf Eliyahu & Einat Edelstien

## Creating Dataset 

We both love Elthon Joan songs, so we thought that it will be great to learn more about the lyrics of his songs.

![alt text](http://musictour.eu/data//uploads/media/albums/769/ab3e7b4194b6ec7a7dadcf43a6040eb3.jpg?raw=true)

### Collecting the Data

We search in all kinds of free lyrics web like: http://www.azlyrics.com/ and created a csv file named 'elton_john_songs_lyrics.csv' which contains all the lines of Elton john songs.

We went through each and every line of song that Elton wrote and we put each line in a different line in the csv file.

The csv file 'elton_john_songs_lyrics.csv' contained 9206 lines from Elton john song’s lyrics.

The csv file 'elton_john_songs_lyrics.csv' is our Dataset.

## Preprocessing

Before we train our model, in the preprocessing stage, we count the unique words.

We will use it later to create a mapping between words and indices, index_to_word, and word_to_index when we create the X and Y vectors of indices. These vectors are the input to the RNN. 

The following code is counting the unique words in the all songs of althon john.


```python
file_to_count = open('elton_john_songs_lyrics.csv').read()
split_file = file_to_count.split()
unique_words = set(w.lower() for w in split_file)
unique_words_len = len(unique_words)
print('The number of unique words in elton john songs lyrics is ' , unique_words_len)
```

    The number of unique words in elton john songs lyrics is  6903
    

In the preprocessing stage we do the following actions:

1. We first read all the sentences from the file 'elton_john.csv', and append a special SENTENCE_START token in the begining of the sentence and append a special SENTENCE_END token to each sentence. This will allow us to learn which words tend start and end a sentence.

2. We tokenize the sentences into words.

3. We count the word frequencies.

4. We get the most common words and build index_to_word and word_to_index vectors. We create a mapping between words and indices, index_to_word, and word_to_index.

5. We replace all words not in our vocabulary with the unknown token.

6. We create the training data. We create the X and Y vectors of indices. The Y vector is used to predict the next word, so the Y is just the X vector shifted by one position with the last element being the SENTENCE_END token.
These vectors are the input to the RNN.

We then use the Theano to utilize the GPU for the SGD Step.



```python
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '5000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '20'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
#print "Reading CSV file..."
with open('data/elton_john.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print ("Parsed %d sentences." % (len(sentences)))
    
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
#print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

#print "Using vocabulary size %d." % vocabulary_size
#print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
t2 = time.time()
print ("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))

if _MODEL_FILE != None:
    load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

num_sentences = 500
senten_min_length = 6

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print (" ".join(sent))

```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-67-a74707f59286> in <module>()
         26 with open('data/elton_john.csv', 'rb') as f:
         27     reader = csv.reader(f, skipinitialspace=True)
    ---> 28     reader.next()
         29     # Split full comments into sentences
         30     sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    

    AttributeError: '_csv.reader' object has no attribute 'next'


## Training Data

Here we start to train our model. The function 'train_with_sgd' is the function which responsible for training the model.
The inputs for the function are: 

- model: The RNN model
- X_train
- y_train
- nepoch: Number of times to iterate through the complete dataset
- learning_rate: Initial learning rate for SGD


```python
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        print()
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print ("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./data/rnn-theano-elton-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

        print()
```

## Generating new stentences
We generate new 500 sentences with at least 6 word in each sentence and write the sentences to txt file named 'model_ejohn_lyrics.txt'.


```python
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str
```

## The RNNTheano 


```python
import numpy as np
#import theano as theano
#import theano.tensor as T
from utils import *
import operator

class RNNTheano:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))      
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.U, self.U - learning_rate * dU),
                              (self.V, self.V - learning_rate * dV),
                              (self.W, self.W - learning_rate * dW)])
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)   


def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print ("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print ("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                print ("+h Loss: %f" % gradplus)
                print ("-h Loss: %f" % gradminus)
                print ("Estimated_gradient: %f" % estimated_gradient)
                print ("Backpropagation gradient: %f" % backprop_gradient)
                print ("Relative Error: %f" % relative_error)
                return 
            it.iternext()
        print ("Gradient check for parameter %s passed." % (pname))
```


```python
import numpy as np

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print ("Saved model parameters to %s." % outfile)
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print ("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))
```

## Finding the Similarity

The following code compare between the original sentence and the generated sentences. 

We use cosine_similarity function to find the similarity between the original sentence and the generated sentences.

The txt file 'original_ejohn_lyrics.txt' contains the original sentences of elton john songs.

The txt file 'model_ejohn_lyrics.txt' contains the generated sentences created by the model. 


```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# original_ejohn_lyrics

original_ejohn_lyrics_sentences = open('original_ejohn_lyrics.txt','r').read().split('\n')
#print(original_ejohn_lyrics_sentences)

# model_ejohn_lyrics

model_ejohn_lyrics_sentences = open('model_ejohn_lyrics.txt','r').read().split('\n')
#print(model_ejohn_lyrics_sentences)


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
```

    empty in again as the open sunlight
    Gonna miss the sunlight
     
    heart be make n't a n't diamante n't time
    And rather all this than those diamante lovers
     
    n't gone to do your little your was
    I was here and I was gone
     
    oh dolly n't everything of night
    In and out of everything
     
    vast believed , up to now
    To think that I believed in you at all
     
    motor in the strength of night dead suzie town
    That keeps my motor running
     
    in a this his your mathematics
    The simple mathematics making up the map
     
    blue ever to start to love
    I love blue eyes
     
    smile can better in i n't passing
    With a smile
     
    they in n't and my woods
    Should I make my way out of my home in the woods"
     
    reckless wire i time your moving
    Left me reckless and abandoned
     
    black lower in n't n't clear
    You cook much better on a lower flame
     
    it n't cotton 'll so n't
    From the times of King Cotton
     
    come all n't are of else
    When are you gonna come down
     
    some freefalling oh in the running
    I'm freefalling into a dream
     
    sing in n't done n't paper
    Sing it
     
    beer n't n't his tonight reads
    Tonight
     
    restless on , out to go
    Go go
     
    no is with go for texarkana
    Just about a mile from Texarkana
     
    and your i n't the too man to worn
    'Cause our love's all worn out
     
    more n't a fire i day
    Day to day
     
    the lightning in is n't free
    I am the thunder, and you are my lightning
     
    on like to name your end to be town
    From the end of the world to your town
     
    across daddy n't paradise to time
    When we return to paradise
     
    meal of the 'm n't side
    Sitting here side by side
     
    that our n't human 's thread
    Watching our love hang by a thread
     
    dirty , the this n't the did
    Take this dirty water
     
    the ceilings try to a left
    From the ceilings of a hundred hotel rooms
     
    when the see n't cotton know
    From the times of King Cotton
     
    na tired n't in dying to in much vain
    I've lived my life in vain
     
    oh in just through n't call
    Just you call, I'll hear
     
    foreman in n't n't that show
    But the foreman he don't worry
     
    an falls n't i room eye
    Can't deny eye for eye
     
    and riding from the parking ,
    And riding on this train
     
    postage time to wanted n't under and dried
    But I'm dried up and sick to death of love
     
    jamaica down n't n't better in this
    In Jamaica all day
     
    in the crawled it to sorry
    And I'd like to say I'm sorry
     
    fog 're to way to n't done
    What you done to me
     
    makes fairground to river all whispers
    With whispers, whispers, whispering whispers
     
    me to through came n't rough
    Where the rain came through
     
    winds so a blow my little door
    Picking up my pain from door to door
     
    bravado in a endless of fire n't plenty
    In a town of plenty
     
    lord your n't to been day
    Day to day
     
    lingers and rain again 'm between for
    In the English rain again
     
    congregation you n't tuned lady to over
    Oh the congregation gathered
     
    dance on n't here n't born take
    When you see me dance
     
    bleeds n't sinner my the wanting cantina
    I'm a natural sinner, born a sinner's son
     
    hoops star and wasteland of hours
    After hours and hours
     
    those ditch christmas to been gammon
    I won't see you `till Christmas
     
    cares is what for to dance
    But January is the month that cares
     
    adventure yi in n't thing his cute
    So it's Ki yi yippie yi yi
     
    time god in been can flag
    Dear God, now's the time
     
    momma you 's is party n't up
    Who likes to party
     
    all know down in the what
    You and I know what it's like
     
    through fake in the waited in from
    Are you fake
     
    knew on he like 's late
    It's late, too late
     
    blue in the ticking of all
    Ticking, ticking
     
    tonight on a n't have no
    Tonight
     
    'd my your came a no night
    I came back for you
     
    wanted so n't n't n't quicksand
    In quicksand
     
    anymore n't no all a do
    Do-do-do-do-do do doo, oh yeah
     
    morning n't door of your keen
    Picking up my pain from door to door
     
    moonshine is by in n't need
    Down by the river where we share a little loving in the moonshine
     
    oh you n't the family of in ways
    And the ways of the world are a blessing
     
    on the every sight 's all
    But something was just out of sight
     
    philosophy like as clover was razors
    We're always happy, __ yeah that's our philosophy
     
    'd your fight on in breath
    I can't breath
     
    in love man is to own
    Is this love
     
    lend for n't i take day
    Dear God, lend a hand
     
    'm your all of your back
    I'm watching all of your secrets
     
    love release n't close to n't
    I'm close to tears
     
    lives four a short to bottles
    How we killed off the bottles
     
    sleep on my and a settles
    Dust settles on a thin cloud
     
    love in , n't know game
    But it's all in the game
     
    move you n't at to dance
    When you see me dance
     
    'm 's monkey in a half immune
    It's always half and half
     
    've stand the mirror this with
    I saw a lie in the mirror this morning
     
    cushy your love to use today
    I got home today
     
    can years in girl and night
    Can tear away the memory of last night's girl
     
    sad buried to no to old
    It's sad, so sad
     
    in my everything all 's busy
    I was everything and nothing all in one
     
    flags have n't god through to willing
    What we need are willing hands
     
    inch to still stood 's dipper
    Big dipper
     
    oh that 's , all time
    I'm waiting all the time
     
    jamaica in the was on away
    Come on Jamaica
     
    have cook for i na signals
    But I think you've got your signals crossed
     
    i daytime i n't good n't tv all dance
    It's daytime in L.A.
     
    decline comes to n't town of soho
    And who could you call your friends down in Soho
     
    grin her n't launch n't n't
    It's hard to grin and bear
     
    , more of ahead as acrobat
    I'm running ahead of my days
     
    from heart for lives n't understand 's
    And not understand. 
     
    boys could a evil to movie
    Yea I've seen your movie
     
    every your lonely again n't there
    Oh I was lonely
     
    row , with she wrote all
    He wrote -
     
    on on n't n't by out
    On and on
     
    all lovers us you your own
    And I'm all on my own
     
    oh from n't you n't n't weeded
    Oh it's a good heart from me to you
     
    house she so space reaching her
    Reaching out, reaching in
     
    no 's it n't in n't west again sea
    Gonna go west to the sea
     
    changing in wins dance to hardwood
    And in the end nobody wins
     
    n't babe n't love n't your n't
    Oh babe, it's the only way
     
    your n't had goes the dove
    Hear the mating call of the morning dove
     
    face on 's to from to hear drunk
    Drunk all the time
     
    where for a get 's n't feeling
    I got a feeling
     
    and 've to they the wine
    On elderberry wine
     
    taste n't 's 's their n't saint
    My baby's a saint
     
    achievement n't my i hold on
    It's a natural achievement
     
    ' and kill to should 's through
    And I guess they're out to kill
     
    loves earth n't out n't had
    Your baby loves loving, my baby loves loving
     
    never got girls washed lyrics n't gown
    Girls, girls, girls
     
    all my her of expensive ,
    Got a digital mind and expensive breath
     
    dusk for on na a keep
    See the dawn come and the dusk hang
     
    blue being n't alone to funky
    Living with her funky family
     
    'm a n't as n't to heading
    Heading for the West
     
    will for this n't line rhyme
    I will see
     
    to 'm n't your n't a riders
    They're looking for more riders
     
    di cost to tricks , criticize
    Would they criticize behind my back
     
    holding a this in day do
    You're holding back
     
    'll a eyes of in the another
    In your eyes
     
    bye as 's n't n't n't dance
    Say bye, bye, bye
     
    and the red n't your hard n't your empty
    And if your heart is empty
     
    surprise to so hard to strength
    Hard to swallow, hard to kill, hard to understand
     
    oh 's n't less in your through
    One less hallelujah
     
    i around a against my train
    To rage against the day
     
    got ever n't `cause from your little juvenile
    Why I'm a juvenile delinquent
     
    blue our is more from late
    It's late, too late
     
    road blue i i n't that
    That beautiful blue, blue sky
     
    pursuing control n't lads in love
    You're pursuing your convictions
     
    friday n't in your nave stumbles
    And it's a wild Friday night
     
    sweet are coloured to time to all ,
    It's my time to write, it's your time to call
     
    stand to from to from love
    To my final stand
     
    spine as is at n't and one
    Running up and down my spine
     
    oh make the rain of sun
    You make the sun shine on me
     
    logs found n't sun my ever
    Logs on the fire
     
    all fifty on n't your much
    I was thirty, I look like fifty
     
    feel inside n't eyes and little
    Inside your eyes
     
    up you i weight 's he
    You take the weight off me
     
    talk big at so n't away
    I'm so
     
    face all n't your your futures star
    When I see your face
     
    and kids n't are 's burns
    When jealousy burns
     
    eyes too n't n't the all
    Now it's all too late
     
    hunters to love on is to love
    Is this love
     
    turned roll in , n't okay you to
    But that's okay
     
    oh much is little your glass
    Oh little Jeannie, you got so much love, little Jeannie
     
    sunrise you n't no n't your to world
    I'm on the road, sunrise is at my back
     
    can acquainted of be war to weakness
    There's a wonder in this weakness
     
    that on the us to rain
    That laughed in the rain
     
    up your n't in your heat
    In your eyes
     
    weighs is in you i get
    When the weight of angels weighs you down
     
    can him n't the take tonight
    They can't take him
     
    di lots last n't 's and now
    But they sure got lots to say
     
    applauded we n't n't n't blue for 's event
    That applauded whenever you laughed
     
    fifteen on n't n't my everything out
    In and out of everything
     
    thrill n't really to girl last
    He's the hoax behind the thrill
     
    was being that n't n't play
    Being used
     
    around in you n't your rain on so door
    Picking up my pain from door to door
     
    us just your your 's misery
    And just like us
     
    i make we n't with before
    We've heard it before, 
     
    brain on your best to light
    And they whispered into your brain
     
    arrested scar it for highway it
    Love leaves a scar
     
    it is tune like wall us
    And just like us
     
    idol 's n't n't i freak
    "No! The kid's a freak"
     
    apartment in around of shakey done
    In that upstairs apartment
     
    real you in to like than
    And is it real
     
    candlelight is makes , n't dream
    I dream of you
     
    man thing in a is or
    Is it day or is it night
     
    trial on the hoop of n't
    If you feel that it's real I'm on trial
     
    chance the over 's like on
    Over and over and over
     
    strange the n't and to game
    And now it all looks strange
     
    born part to curls n't no
    To straighten out them curls
     
    two n't in n't that and
    With me and you it's two by two just like Noah's ark
     
    blue unmade in the golden devil
    The devil in a suit
     
    travelling coach i feet n't in a tree
    Round a tree in the summer
     
    true from love in loudly dance
    True love, true love
     
    found and part in your her to girl
    To be a part of
     
    i die n't i your made
    And they made you change your name
     
    way my bravado of see away
    But see me once and see the way I feel
     
    we better n't the someday eagle
    Someday
     
    the tack time in the they early you
    I found the calling early
     
    room a you we 's in who
    Sitting in my room
     
    going everyone for him to seems
    Me and you we're not for everyone
     
    wake for down , n't n't of n't had
    I don't want to wake you
     
    flows secrets heading n't from twenty
    The secrets that you keep
     
    and hear in way your play
    We hear, we hear your name
     
    cup to your n't it n't side
    But I'll drop it in your tin cup
     
    and is to you the vain
    I've lived my life in vain
     
    sticking 'till n't me n't make
    I got feet sticking out of my shoes
     
    never heat to you on another
    Another nickel on another name
     
    rumor in the take you one
    There's a rumor of a war that's yet to come
     
    to a wreck n't 's to swinging , waiting
    I'm waiting, I'm waiting for
     
    word not by the stand on
    Maybe if I promise not to say a word
     
    miles n't you n't over 's 'd n't
    Over and over and over
     
    have belief in n't a 'm girl
    Man stumbles on his own belief
     
    social try hear all so your way
    There's not much I don't hear of and still you try
     
    been 's vast n't from 's cared
    He must have been a gardener that cared a lot
     
    all , and instead has for
    Close to you instead
     
    on was in n't your time
    There was a time
     
    to wire in is was story
    We all know the story, 
     
    so n't barking in a somebody
    My bulldog is barking in the backyard
     
    cozy heaven is late to into
    It's late, too late
     
    make you it in this touching
    Touching truth and touching skin
     
    brian and no never n't governor
    Johnny and the governor
     
    , dove n't from i take our the greek
    It's all just Greek to me
     
    graveyard is your go in dead
    To have a graveyard as a friend
     
    is i for on n't the 're
    On and on
     
    used for love of 's for situation to deeper
    It's burning much deeper this time
     
    blue into n't with your more
    Just look into that beautiful blue
     
    i you right on n't be my denying n't
    And there ain't no use denying
     
    protection day in the distant god
    The best of all protection
     
    forever on we n't n't light
    And when we start we say forever
     
    already in n't neanderthal n't beast
    I'm a neanderthal man
     
    bad on 'd n't worse n't game
    For better or for worse
     
    could in i big texas boy
    `Cause she hates those Texas girls
     
    enough in an for your bragging
    Now I hear you're bragging one is not enough
     
    sandwich , n't across we past
    All across the world
     
    SENTENCE_START of have my symbols lay
    Where the symbols painted black and white
     
    you water in here to dye
    Like red dye in my veins
     
    blue are n't time lights pretty
    On a blue blue day
     
    drug to c'est your was it
    It's like some drug for you
     
    on she man the apartment n't
    In that upstairs apartment
     
    n't love to wild to still
    To see if you still love me
     
    time of town n't so to wed
    Got a reason why they shouldn't wed
     
    sluice love n't sheets and certain
    I have loved every sluice in your harbor
     
    hill in a want dress 's n't grown 's tied
    Somewhere on the hill
     
    of it in a born na one
    Born to lose
     
    unforgiven never big 're ! n't racing
    I go racing back like a man possessed
     
    her chalk to found n't what
    Chalk up one more crazy notion
     
    demon we little on your stars
    If only we were stars, you and I
     
    nowhere sweet the star is filled
    It came out of nowhere
     
    'll with n't falls so change
    When the snow falls
     
    gun , nights n't your feathers
    After claws and feathers
     
    desire of find n't their basement
    `Cause I miss my basement
     
    left is stately and me disturb
    Don't disturb me if you dare 
     
    do to your no heart 's
    Do-do-do-do-do do doo, oh yeah
     
    heaven , n't n't touch n't shacked 's own
    He's shacked up in his basement making P.C.P.
     
    all is sweet n't my way
    Life is oh so sweet
     
    no rouge 's no the names
    No no no no no no she don't like to fight it
     
    oh with graves you to blast
    To hear at last the blast of doom
     
    baby blue is your season is on
    It's open season
     
    these choose n't that hands to our
    These words
     
    'll know what n't little time
    And I don't know what the time is
     
    howl weeds n't with n't do
    The lemons and weeds
     
    lips as be intend to deep
    Did we intend
     
    she on me 's n't above
    Above the people
     
    on to ferro that n't low
    You know I'm too low, too low, too low for zero
     
    off dirty and 're n't after 's last
    You're dirty, but you're worth it
     
    shacked the you to hands to us
    He's shacked up in his basement making P.C.P.
     
    bye would in the o'clock happy
    Say bye, bye, bye
     
    very your true in i ice
    If it's true I'm in your hands
     
    i are is they your lake
    There are many graves by a cold lake
     
    tonight around those in an those
    Those that flew and those that fell
     
    pre-flight man n't the your dead
    She packed my bags last night pre-flight
     
    deed to a understand ironing blue
    And so the deed is done
     
    on day in you is electricity
    Day to day
     
    oh go in for n't honky you
    You better get back honky cat
     
    ceiling for your ca wed your
    No ceiling on hard living
     
    never a chloe n't on give
    Chloe
     
    will you i our or up n't dance off you
    When you see me dance
     
    waiting and n't cold to screaming
    They've all heard me screaming, screaming
     
    and is n't 's acting you
    The acting right, not acting up
     
    not na they of n't in
    I'm not alone in the night
     
    rigid through n't n't be day
    But the thing that shook her rigid
     
    creek you in your an muscle
    Out a dried up creek
     
    idolise in 's the his fight
    And idolise twisted cars
     
    that already to fall to go
    What we already know
     
    made their it so n't know
    I'm so
     
    gambler while , take in yeah
    Yeah, Yeah, oh Yeah
     
    just all n't sleep 's night
    I just drink myself to sleep each night
     
    july troubled the i what in us
    It's July but it's cold as Christmas
     
    had your n't n't in 's
    You had a party raging in your head
     
    disguise my 're n't n't heart
    You're a mystery of disguise
     
    blue on n't mining in say
    On a blue blue day
     
    , bruise in and find na burns
    They say we bruise too easily 
     
    for more hope n't one take no
    No more singing duets for one
     
    just i everything 's points you
    All points east, west and in-between
     
    colonel is have is on down
    Every day I watch the colonel smile
     
    on 's a toast of for
    For there's toast and honey
     
    separate on i 's love before
    In two separate world
     
    rules from do n't n't dream
    You get to make the rules
     
    thing and n't on this night
    And I'll go on and on this way
     
    things 's been n't know so you
    I know you and you know me
     
    blue is n't side is could
    And time is never really on my side
     
    shirts forever n't were first cold
    The white shirts in the moonlight
     
    on hand 's in an juvenile
    Passed on from hand to hand
     
    's things n't wrap of today to confront
    I'd claim a confront
     
    find on 's row 's to she
    Landing on skid row
     
    bold floors on i so than
    From dirt and damp to hardwood floors
     
    land 're n't writing n't lure
    It was the lure of the tropics
     
    see with roses in he n't see
    I will see
     
    you madman the you 's of all
    Take my word I'm a madman don't you know
     
    crocodile cost blue and the n't
    But they'll let us know the cost
     
    fog for my 've found been
    I've been searching for you all my life
     
    so in cellophane on your us
    The cellophane still on the flowers
     
    across life to street of crazy man
    On the street
     
    fragrance , that , now home
    The fragrance of you on the wind
     
    came in 's happy impressa you
    S'erra solo impressa in me
     
    tired down to a while too
    I'm too tired to work
     
    flutes to wo to hacking 's home
    Baccarat and champagne flutes
     
    it go in me dive old
    I dive in, I dive deep, I just swim
     
    shake that on to little heart
    When you need a little shake
     
    about back end i their butterfly
    You're a butterfly
     
    could hercules to named never into
    A cat named Hercules
     
    wedding wrote to n't for your very
    He wrote -
     
    i new learn n't learn n't so n't
    Another you learn, nothing
     
    hunted part n't n't n't change come
    The hunter and the hunted
     
    state n't , of 're in 'll
    In the state of illusion
     
    have fun 's better n't tell
    Gonna have some fun
     
    glory can summer that n't time
    If you're looking for the glory
     
    love of out of love of true
    True love, true love
     
    rhythm n't to n't n't 've
    By the rhythm of the rain
     
    i hold to below at are
    To have and to hold
     
    there five and i woman oh
    Oh don't believe that woman please
     
    still ! n't in get again
    Time and again I get ashamed
     
    are on your how to body 's in over
    Over and over and over
     
    across in the lost of crowd
    Into the crowd
     
    past back it n't the exchange
    In exchange for the sweetest addiction
     
    it n't wits n't oh hope
    Hope we're gonna make it
     
    loosen , n't the our songs
    Loosen your lips
     
    like through on to those n't hotels
    Drunken nights in dark hotels
     
    glory hears 's last have help
    When help at last arrives
     
    idolise with so your time i way
    And idolise twisted cars
     
    have won strong at love cheat
    I never wanted to cheat
     
    some same is on the dolly tonight
    Tonight
     
    stay 's n't on the evil
    For you to stay
     
    gathered wild men in a squeezing
    Oh the congregation gathered
     
    holes hearts a shallow her n't n't my l.a. is
    A shallow heart that left her cold
     
    tonight about on on n't weak i sing again
    On and on
     
    young to got no so foresee
    I guess that it's mine for foresee
     
    is it in one a not
    Now I hear you're bragging one is not enough
     
    blue love is an n't the dusk
    I love blue eyes
     
    is roll you the your clovers in n't to victim
    Victim of love, victim of love
     
    day is a snake the grind
    And snake hips Joe is Mr. Cool
     
    wrangler heard n't your school down
    And a heart out west in a Wrangler shirt
     
    might look to cavaliers of side
    You might still look pretty
     
    through the flown to your cue 's come
    The lights go down on cue
     
    thing under your been n't to we
    And so say your lovers from under the flowers
     
    was for 's a little home when
    When I was down
     
    us is in n't eye ,
    Can't deny eye for eye
     
    with n't love , your ]
    With you my love
     
    it miss , na n't time
    I miss the earth so much I miss my wife
     
    red iron n't be fantastic take were
    Fantastic the feedback
     
    old-fashioned i action of awakening 's rooms
    Get a little action in
     
    boxer , 've an our take
    He could have been a boxer
     
    reign on my the sky of good
    Of the Transvaal sky
     
    know n't n't the da n't your n't death
    Da da da-da-da-da-da da da-da da
     
    this the 've rewarded your gone man
    Gone they've gone away
     
    like in prove we your taunts
    As temptation taunts the fox
     
    oh been in n't right vermin
    Burning vermin stink
     
    keep gifted in into with be
    To be young, gifted and black
     
    jeans is a and the love
    Is this love
     
    to no 's rain to got
    Got no time to lose
     
    tonight on a loosen next one
    Loosen your lips
     
    two-lane on trust prose in been jeans
    A two-lane highway winding past a desert town
     
    make my n't n't on you
    You make the sun shine on me
     
    from drain of without to eyes
    Without love
     
    spoke is 's two-fister n't the no inside
    Soul sister, two-fister
     
    cold for i trial 's few
    Cold cold heart
     
    heart and n't that is your screens
    And if your heart is empty
     
    was your n't this na all
    And I was you
     
    holds , the sidewalk i loving
    Leaner on the sidewalk
     
    knew n't da-da in your nowhere
    Da da da-da-da-da-da da da-da da
     
    cares in the altar-bound for jeannie
    Altar-bound, hypnotized
     
    sting soul on on i only
    On and on
     
    take it n't a answer way
    Where is the answer
     
    can yell with faithful loosens were
    So I better yell help
     
    n't really n't fool n't out
    You play me for a fool
     
    paint got n't i butterfly fever
    You're a butterfly
     
    in you on wan of take
    You can take her
     
    out on like 's n't who
    Who leads who, who knows where
     
    time is n't that to your time
    It's my time to write, it's your time to call
     
    ladder handle n't speed my puppet
    I'm your puppet, I'm your puppet
     
    we time and everything in a between
    In and out of everything
     
    see your this 's n't evening
    Talking through the evening
     
    it captives on , 's articles
    Say goodbye to articles on who the senator kissed
     
    we sailors n't never day night
    Day to day
     
    oh what in the an justified
    I was justified when I was five
     
    really n't n't go your dance
    Go go
     
    crime to n't n't n't turned
    Crime in the streets
     
    found on with in your squirrels
    Deep in the woods the squirrels are out today
     
    wins so door for 's never n't you
    No one ever wins
     
    reckless for and n't you to my diamonds
    Left me reckless and abandoned
     
    and friend n't their my touch
    Just feel their gentle touch
     
    steps n't been out n't first
    From the first kiss
     
    devil-dark , n't sun in tonight
    Stumbling through the devil-dark
     
    convenience are , on 's to water
    Crazy water
     
    town to were and just in out
    And I just can't wait to get out of this one horse town
     
    governor on in know to your care
    Johnny and the governor
     
    to at n't n't n't only the wide-eyed
    That the wide-eyed and laughing
     
    looking for his shine in the shall
    Shine a light shine a light
     
    would what n't n't to flight
    What would you do
     
    hell on n't n't n't understand down
    I understand I'm on the road
     
    make out to your up in the captives
    We're just captives in our separate cells
     
    brothers and be to get here
    Here and there
     
    wife to n't 's her here
    Without a wife in line
     
    what roses of the a gon
    The roses in the window box
     
    troubles a ten out and times
    Nine times out of ten
     
    been big everything in the endless so for
    Endless nights on an endless sea
     
    'm wind of n't last is stars
    The last I heard of you
     
    oh on in your eyes and world
    In your eyes
     
    love and n't causes on ring fits
    Dying for causes
     
    god on on the na too
    On and on
     
    church from bone arrow 's more
    One more arrow
     
    are get i empty n't is
    And if your heart is empty
     
    at the are n't have your president
    And kids still respected the president's name
     
    have spawn 's the crown stretch
    I'm chasing the crown, the crown, I'm chasing the crown
     
    two i hold under daggers , n't gather feeling 's singer
    I got a feeling
     
    go is n't look n't know
    Go go
     
    're on 's rain 's at the high
    On high, with nothing to do
     
    blue of the fantasy of her
    I play my games of fantasy
     
    talk n't the need 's from
    Down on the jive talk
     
    mating in like n't describe to try
    And I can't describe
     
    baby on in the need n't side
    Sitting here side by side
     
    street i a catwalks to the washing
    On the street
     
    them is n't n't in in
    Put them in a box somehere, put them in a drawer
     
    sinking to so n't 's book
    And I read it in your book
     
    wait is 've n't i the wonderful
    I've got to wait till morning
     
    love of the man of are
    She's the love of your life
     
    so lightning n't dentro and time
    I am the thunder, and you are my lightning
     
    youth to ai is your songs
    Your songs have all the hooks
     
    stare the again 're her glue
    And if you stare into the darkness
     
    and your make in to back again
    Back into my bed again
     
    days in the hare with wide
    But when you're turtlesque, I'm a hare's breath
     
    heights dry-docked in in 're n't night
    I feel I'm dry-docked and tongue-tied
     
    my time n't fire 's her hero
    Emily prays to a faded hero
     
    hundred i be so my her
    There must have been a hundred
     
    wrote is n't fleet n't age
    He wrote -
     
    around it over like n't give
    Over and over and over
     
    by your up n't make time
    Make your mind up fast
     
    oh see in n't name time
    In the name of you
     
    on 's love in the coat
    Pull my coat around me
     
    were to so everything to dance
    I was desperate to dance
     
    little 're n't to city felt
    City boy, city girl
     
    sent mutual in your 's between for his in 's cock
    Mutual misunderstanding
     
    information n't a guards in here in more
    Help me information
     
    fool in the confidence it connections
    I got connections with the underground
     
    so n't they what so upon
    I'm so
     
    where all is n't i time
    Where all that was is gone
     
    from more n't n't you to hope
    It's more than I can take from you
     
    in 'm n't her hear thing
    Come down in time I still hear her say
     
    wayne place through to time girl
    I've got John Wayne stances
     
    ticked , about on your your from
    Neither of us understood the way things ticked in Hollywood
     
    time in n't to n't to up
    It's my time to write, it's your time to call
     
    use whores n't design on burns
    Of their powerful design
     
    again your sad in home enough
    It's sad, so sad
     
    palace for you all n't as
    Like the roof of a palace
     
    n't saturday to another your so
    Another ride, another tune
     
    island love to time tag there
    Island girl
     
    on is so no your it
    No no no no no no she don't like to fight it
     
    plough to 'm you , hear
    I'm going back to my plough
     
    las in a every for in
    Every place, every way
     
    doubt simple n't empty 's fall
    Just a simple word
     
    ever years with , line know
    But with the passing of the years
     
    sand , get n't good to us
    Some of us never get to see
     
    to talks n't the wait ignorance
    To ignorance and innocence
     
    while on n't with your know
    And while I'm away 
     
    snow in me 's enough in slowly
    The fruit juice flowing slowly, slowly, slowly
     
    closed up to n't brand-new n't your my like
    This is your brand new brother
     
    we soon a meets n't again
    Swears the South's gonna rise again soon
     
    heels to 're is is one
    But one is all we need
     
    man heaven your have cow heart
    Poor cow
     
    wrong it filled is room up
    Keeping it from you is wrong
     
    is it a brought a prima dance
    How you brought what's inside out
     
    he you n't a whip like on
    And I like you like that
     
    i line in what to faces
    Fall in pieces from our faces
     
    doubt in the teas n't places
    I'm lost in doubt and I'm all alone
     
    life alive n't n't meant out
    Life
     
    player kool n't your and in n't hold hitch
    A player just acting for me
     
    oh we n't mellow to brussels
    You make me mellow, I make you mellow
     
    n't for man to the dignity
    I've seen dignity fail and colours run
     
    garretts a could that n't sartorial
    You've a certain sartorial eloquence
     
    kindly will religion n't your wheel
    Smiled back at us kindly
     
    rest in from n't new eyelid
    How it's laid to rest
     
    meko is n't nothing n't n't eyes
    But I liked the Meko
     
    while for i all n't starts
    Tell me where living starts
     
    tunes would n't n't resting n't their
    His black hands resting on the keys
     
    's got to that , tell
    But I'd like to tell you that I love you
     
    from something to another a so light
    I'm seeing life in another light
     
    you notre on smell and daughters card
    Like the ancient bells of Notre Dame
     
    shines is n't did to love 's
    And my love still shines, shines on you
     
    baby to my he up all
    My baby's a saint
     
    you waiting of sentimental of he
    What's he waiting for
     
    into in who wildcat n't bass
    Look at me twice with wildcat eyes
     
    this got in do with hands
    With the things you do
     
    clock better n't to save go
    Go go
     
    sister of i too for sleeping
    And I'm too old for you
     
    their go forever n't a threw know
    You know you can't hold me forever
     
    no kingdom to limousine n't darn
    Everybody's kingdom must end
     
    he n't 're history on n't grave
    Everything is history
     
    immune the party time your path
    The path of time
     
    am world at 's manners her
    When manners make no difference
     
    SENTENCE_START for a hurry to rain in angeline
    And trust me, Angeline
     
    letter in 're heard to shoes
    I have to write a letter
     
    his drifting on in to all
    I'm drifting in your hoodoo
     
    can cloud and n't before again
    There was a steel cloud
     
    live so it how n't all
    How you handle what you live through
     
    stairs in the hurt 's tomorrow
    There's no tomorrow, 
     
    is i n't up out ,
    When your back's up, sweat it out
     
    that for ? on your overtime
    She'll work a little overtime
     
    down only so with the one up
    She's the only one, makes me feel so good
     
    married are a dance that 's
    Of each married man
     
    The similarity between the original and model is  0.000895480331703
    

Here you can see the original sentence and the most similar generated sentence.

We can learn on the similarity between the original sentences of elton john songs and the generated sentences created by the model.

![alt text](http://i.telegraph.co.uk/multimedia/archive/02660/eltonJohn_2660332b.jpg?raw=true)

## Conclusion

This was a very intensive project and we learn a lot!
We had to learn a new language (python) and combine all the tools we acquired in the class within it.
As we mentioned before, we wanted to test and see if a neuron network can learn sentence from songs specifically Elton Joan songs which we really love.

As you can see the similarity index in our result was not higher only 0.0009, but this can be explained by the following reasons.
We conclude that the reason behind it is because we didn't run the RNN with a lot of iterations. You can see in the code above that the variable "NEPOCH" equals only 20.

At the beginning we tried to run the RNN while the variable "NEPOCH" equals to 5 and to 10 until we stopped at 20. We saw that as this variable is getting bigger the similarity index is getting higher and we receive better results. 

The reason we didn’t run the RNN with a lot of iterations (higher than 20) was mainly hardware. Our computer are not strong enough and several time we had blue screens and the CPU was sky rocketing. (It was steady on 98-100% even with only 20 runs and it took a lot of time also).
The hardware in the university were even lower than our computers.

Another issue was the graphic card drivers. Using THEANO the right way meaning using GPU but we couldn't do it. we tried several drivers but nothing change still the CPU utilization was quite high.
This is a common problem with this package and after reading a lot online we figure out our graphic cards aren't suitable.

Another way we try to receive a better results was by running the code in amazon but it didn't work. The configuration was a mess and we lack the experience working with it. 

We also learned that a change in the size of the vocabulary does effect the results we get and the similarity index. Although we had poor result, the limited size of the vocabulary (about 5000 words) gives us mitigated performance. 
We had to limit the size of words in order to have a better results.

We learned that as the size of the vocabulary get bigger, the time to learn is growing and vice versa.

We think that another reason for the poor results is the data we created - we focused on lines from songs. Each line in the CSV file represent a line from a song, but this is not a homogeneous data.
Is it safe to assume that lines of a song by the same song writer have the same context?
Well after seeing the result the answer might be NO. 
But maybe a good way to test it further is to try finding similarity by those method which we thought about:
1. Test songs within the same album
2. Make each line in the data as a complete song and not only a line from a song which is lack of context.

It was interesting to look at the semantic dependencies of sentences from elton john songs. We enjoyed learning about RNN and machine learning and also creating our own dataset from scratch.   


```python

```
