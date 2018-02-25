import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    # From 0 to p-t
    for i in range(len(series) - window_size):  
        inp = []
        
        # Pick next window_size elems
        for j in range(i, i + window_size):
            inp.append(series[j])
            
        # Add input and output
        X.append(inp)
        y.append(series[i+window_size])
        
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # run your model!
    model.fit(X_train, y_train, epochs=1000, batch_size=50, verbose=0)

    return model



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import string
    alphabet = string.ascii_lowercase + ' '
    punctuation = ['!', ',', '.', ':', ';', '?']
    keepchars = alphabet + ''.join(punctuation)
    #convert to list and replace unwanted chars with blanks
    listtext = list(text)
    for i,char in enumerate(listtext):
        if char not in keepchars:
            listtext[i] = ' '
    text = ''.join(listtext)
    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    ctr = 0
    
    # Goes from window_size until the end, and pick previous characters
    for i in range(window_size, len(text), step_size):
        inputs.append(text[ctr:i])
        outputs.append(text[i])
        ctr = ctr + step_size
    
    return inputs,outputs

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    from keras.layers import Activation
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation('softmax'))
    return model
