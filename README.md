# Chatbot-for-customer-service
Here's a line-by-line explanation of the code:
1. import nltk
This line imports the Natural Language Toolkit (NLTK) library, which is a popular library for natural language processing tasks.
2. nltk.download('punkt')
This line downloads the Punkt tokenizer models, which are used for tokenizing text into individual words or tokens.
3. nltk.download('wordnet')
This line downloads the WordNet lexical database, which is a large lexical database of English words.
4. from nltk.stem import WordNetLemmatizer
This line imports the WordNetLemmatizer class from the NLTK library. This class is used for lemmatizing words, which means reducing words to their base or root form.
5. lemmatizer = WordNetLemmatizer()
This line creates an instance of the WordNetLemmatizer class.
6. import json
This line imports the JSON (JavaScript Object Notation) library, which is used for working with JSON data.
7. import pickle
This line imports the Pickle library, which is used for serializing and deserializing Python objects.
8. import numpy as np
This line imports the NumPy library, which is a popular library for numerical computing.
9. from keras.models import Sequential
This line imports the Sequential model class from the Keras library, which is a popular library for deep learning.
10. from keras.layers import Dense, Activation, Dropout
This line imports the Dense, Activation, and Dropout layer classes from the Keras library. These layers are used for building neural networks.
11. from keras.optimizers import SGD
This line imports the Stochastic Gradient Descent (SGD) optimizer class from the Keras library. This optimizer is used for training neural networks.
12. import random
This line imports the Random library, which is used for generating random numbers.
13. words = []
This line creates an empty list called words, which will be used to store the words in the training data.
14. classes = []
This line creates an empty list called classes, which will be used to store the classes or labels in the training data.
15. documents = []
This line creates an empty list called documents, which will be used to store the documents or sentences in the training data.
16. ignore_words = ['?', '!']
This line creates a list called ignore_words, which contains the words that should be ignored when processing the text.
17. data_file = open('/content/intents.json').read()
This line opens a file called intents.json and reads its contents into a string called data_file.
18. intents = json.loads(data_file)
This line parses the JSON data in data_file into a Python dictionary called intents.
19. for intent in intents['intents']:
This line starts a loop that iterates over the intents dictionary.
20. for pattern in intent['patterns']:
This line starts a nested loop that iterates over the patterns list in each intent.
21. w = nltk.word_tokenize(pattern)
This line tokenizes the pattern string into individual words using the NLTK tokenizer.
22. words.extend(w)
This line adds the tokenized words to the words list.
23. documents.append((w, intent['tag']))
This line adds a tuple containing the tokenized words and the corresponding class label to the documents list.
24. if intent['tag'] not in classes:
This line checks if the class label is already in the classes list.
25. classes.append(intent['tag'])
This line adds the class label to the classes list if it's not already there.
26. words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
This line lemmatizes the words in the words list and removes any words that are in the ignore_words list.
27. words = sorted(list(set(words)))
This line sorts the words list and removes any duplicates.
28. pickle.dump(words, open('words.pkl', 'wb'))
This line saves the words list to a file called words.pkl using the Pickle library.
29. pickle.dump(classes, open('classes.pkl', 'wb'))
This line saves the classes list to a file called classes.pkl using the Pickle library.
30. training = []
This line creates an empty list called training, which will be used to store the training data.
31. output_empty = [0] * len(classes)
This line creates a list called output_empty that contains zeros for each class label.
32. for doc in documents:
This line starts a loop that iterates over the documents list.
33. bag = []
This line creates an empty list called bag, which will be used to store the bag-of-words representation of each document.
34. pattern_words = doc[0]
This line gets the tokenized words for each document.
35. pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
This line lemmatizes the words in each document.
36. for w in words:
This line starts a nested loop that iterates over the words list.
37. bag.append(1) if w in pattern_words else bag.append(0)
This line adds a 1 to the bag list if the word is in the document, and a 0 otherwise.
38. output_row = list(output_empty)
This line creates a copy of the output_empty list.
39. output_row[classes.index(doc[1])] = 1
This line sets the corresponding class label to 1 in the output_row list.
40. training.append([bag, output_row])
This line adds the bag and output_row lists to the training list.
41. bag_len = len(words)
This line gets the length of the words list.
42. for i in range(len(training)):
This line starts a loop that iterates over the training list.
43. if len(training[i][0])!= bag_len:
This line checks if the length of the bag list is equal to the length of the words list.
44. training[i][0] = training[i][0][:bag_len] + [0] * (bag_len - len(training[i][0]))
This line pads or truncates the bag list to the desired length.
45. training = np.array(training, dtype=object)
This line converts the training list to a NumPy array.
46. train_x = list(training[:, 0])
This line gets the bag lists from the training array.
47. train_y = list(training[:, 1])
This line gets the output_row lists from the training array.
48. print("Training data created")
This line prints a message indicating that the training data has been created.
49. model = Sequential()
This line creates a Sequential model.
50. model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
This line adds a Dense layer with 128 neurons and ReLU activation.
51. model.add(Dropout(0.5))
This line adds a Dropout layer with a dropout rate of 0.5.
52. model.add(Dense(64, activation='relu'))
This line adds another Dense layer with 64 neurons and ReLU activation.
53. model.add(Dropout(0.5))
This line adds another Dropout layer with a dropout rate of 0.5.
54. model.add(Dense(len(train_y[0]), activation='softmax'))
This line adds a final Dense layer with a number of neurons equal to the number of class labels and softmax activation.
55. sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
This line creates an SGD optimizer with a learning rate of 0.01, decay of 1e-6, momentum of 0.9, and Nesterov acceleration.
56. model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
This line compiles the model with categorical cross-entropy loss, the SGD optimizer, and accuracy metrics.
57. hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
This line trains the model on the training data for 200 epochs with a batch size of 5 and verbose output.
58. model.save('chatbot_model.h5', hist)
This line saves the trained model to a file called chatbot_model.h5.
59. print("model created")
This line prints a message indicating that the model has been created.



Next, we will create Python code to build a simple chatbot that uses a pre-trained model to predict the class of a user's input and generates a response based on that class. For this, we need to import Libraries and load models.

Here is a line-by-line explanation of the code:
Importing Libraries
from nltk.stem import WordNetLemmatizer: imports the WordNetLemmatizer class from the NLTK library.
lemmatizer = WordNetLemmatizer(): creates an instance of the WordNetLemmatizer class.
import json: imports the JSON library.
import pickle: imports the Pickle library.
import numpy as np: imports the NumPy library and assigns it the alias np.
from keras.models import load_model: imports the load_model function from the Keras library.

Loading Models and Data
model = load_model('/content/chatbot_model.h5'): loads a pre-trained chatbot model from a file named chatbot_model.h5 in the /content directory.
words = pickle.load(open('/content/words.pkl', 'rb')): loads a list of words from a file named words.pkl in the /content directory using the Pickle library.
classes = pickle.load(open('/content/classes.pkl', 'rb')): loads a list of classes from a file named classes.pkl in the /content directory using the Pickle library.

Defining Functions
def clean_up_sentence(sentence):: defines a function named clean_up_sentence that takes a sentence as input.

sentence_words = nltk.word_tokenize(sentence): tokenizes the input sentence into individual words using the NLTK library.

sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]: lemmatizes each word in the tokenized sentence using the WordNetLemmatizer instance.

return sentence_words: returns the lemmatized sentence words.

def bow(sentence, words, show_details=False):: defines a function named bow that takes a sentence, a list of words, and an optional show_details parameter.

sentence_words = clean_up_sentence(sentence): calls the clean_up_sentence function to lemmatize the input sentence.

bag = [0]*len(words): creates a list of zeros with the same length as the input list of words.

for s in sentence_words:: loops through each word in the lemmatized sentence.

for i,w in enumerate(words):: loops through each word in the input list of words.

if w == s:: checks if the current word in the input list is equal to the current word in the lemmatized sentence.

bag[i] = 1: sets the corresponding index in the bag list to 1 if the word is found.

if show_details: print ("found in bag: %s" % w): prints a message if the show_details parameter is True.

return(np.array(bag)): returns the bag list as a NumPy array.

def predict_class(sentence, model):: defines a function named predict_class that takes a sentence and a model as input.

p = bow(sentence, words,show_details=False): calls the bow function to create a bag-of-words representation of the input sentence.

res = model.predict(np.array([p]))[0]: uses the input model to predict the output probabilities for the input sentence.

ERROR_THRESHOLD = 0.25: sets an error threshold for filtering out low-confidence predictions.

results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]: filters out predictions below the error threshold and creates a list of indices and corresponding probabilities.

results.sort(key=lambda x: x[1], reverse=True): sorts the list of predictions in descending order of probability.

return_list = []: creates an empty list to store the final predictions.

for r in results:: loops through each prediction in the sorted list.

return_list.append({"intent": classes[r[0]], "probability": str(r[1])}): appends a dictionary containing the predicted intent and probability to the return_list.

return return_list: returns the list of predictions.

def getResponse(ints, intents_json):: defines a function named getResponse that takes a list of predictions and a JSON object as input.

tag = ints[0]['intent']: extracts the predicted intent from the input list of predictions.

list_of_intents_tag = intents_json['intents']: extracts a list of intents from the input JSON object.

for i in list_of_intents_tag:: loops through each intent in the list.

if(i['tag']== tag):: checks if the current intent matches the predicted intent.

result = random.choice(i['responses']): selects a random response from the list of responses for the matching intent.

break: breaks out of the loop once a response is found.

return result: returns the selected response.

def chatbot_response(msg):: defines a function named chatbot_response that takes a message as input.

ints = predict_class(msg, model): calls the predict_class function to predict the output probabilities for the input message.

res = getResponse(ints, intents): calls the getResponse function to retrieve a response based on the predicted intent.

return res: returns the retrieved response.

Creating the Chatbot
print("Welcome to the chatbot! Type quit to stop the chat."): prints a welcome message to the user.
while True:: enters an infinite loop to keep the chatbot running.
msg = input(""): prompts the user to enter a message.
if msg.lower() == "quit":: checks if the user wants to quit the chatbot.
break: breaks out of the loop if the user wants to quit.
print(chatbot_response(msg)): calls the chatbot_response function to retrieve a response to the user's message and prints it to the console.



After this, we need to create a Gradio app
Here is a line-by-line explanation of the code:
Installing Gradio
!pip install gradio: This line installs the Gradio library using the pip package manager. The ! symbol is used to run shell commands in a Jupyter notebook.

Importing Gradio
import gradio as gr: This line imports the Gradio library and assigns it the alias gr. This allows us to use the gr prefix to access Gradio's functions and classes.

Defining the Chatbot Function
def chatbot(query):: This line defines a function named chatbot that takes a single argument query. This function will be used to process user input and generate a response.

Defining the Dummy Data
intents = {... }: This line defines a dictionary named intents that maps user queries to responses. The dictionary contains three key-value pairs:
"hello": "Hi, how can I help you?": This pair maps the query "hello" to the response "Hi, how can I help you?".
"goodbye": "Goodbye, have a great day!": This pair maps the query "goodbye" to the response "Goodbye, have a great day!".
"thank you": "You're welcome!": This pair maps the query "thank you" to the response "You're welcome!".

Checking for Intent Matches
for intent, response in intents.items():: This line starts a for loop that iterates over the key-value pairs in the intents dictionary.
if query.lower() == intent:: This line checks if the user's query (converted to lowercase) matches the current intent. If a match is found, the function returns the corresponding response.
return response: This line returns the response associated with the matched intent.

Returning a Default Response
return "I didn't understand that. Can you please rephrase?": If no match is found in the intents dictionary, this line returns a default response indicating that the chatbot didn't understand the user's query.

Creating a Gradio App
app = gr.Interface(... ): This line creates a Gradio app using the gr.Interface function. The app takes a text input and returns a text output.
fn=chatbot: This parameter specifies the function to use for processing user input and generating a response. In this case, it's the chatbot function defined earlier.
inputs="text": This parameter specifies the type of input to expect from the user. In this case, it's a text input.
outputs="text": This parameter specifies the type of output to return to the user. In this case, it's a text output.
title="Customer Service Chatbot": This parameter specifies the title of the app.
description="Ask me anything!": This parameter specifies the description of the app.

Launching the App
app.launch(): This line launches the Gradio app using the launch method. The app will be displayed in a web interface, and users can interact with it by entering text input and receiving responses.
