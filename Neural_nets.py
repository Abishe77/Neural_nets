#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow.keras.datasets import mnist #This tesnorflow is only for dataset 
import numpy as np
import matplotlib.pyplot as plt

#Load the datasets as training and testing samples
(x_train ,y_train),(x_test,y_test) = mnist.load_data()

'''The dataset contains images in 28x28 grid which is 784 pixels so we need to convert them into a long vector
so that it can be matrix multiplied and normalize the pixel values because currently the values are [0,255] where
0 corresponds to black color and 255 corresponds to white color and other inbetween are grey shades 
Normalizing makes the computation faster and easier instead of multiplying with 255 and mormalizing squishes 
the value between [0,1] '''

x_train = x_train.reshape(x_train.shape[0], 784)/255 #Intuition shape[0] = 60000 (number of images) , shape[1] = 28(height) , shape[2] = 28(width)
x_test = x_test.reshape(x_test.shape[0],784)/255


'''
INTUITION BEHIND THE CODE : (Forward pass own logic) , Setup : 2 hidden layers 10 neurons each

 first z = WX + b where X is the input layer consisting of 60000 images and each single image is represented by 784 squares(pixels) so
 the matrix is formulated as (60000 x 784) and for first hidden layer it is 10 neurons so each neurons must scan all 784 pixels to find a number 
 so it is (1 x 784) so for 10 it is ((1 x 10) x 784) =(10 x 784) so thats it now z = WX +b that is (10 x 784) x (60000 x 784) but 
 it will give dimension mismatch so transpose X which is (10 x 784) x (784 x 60000) which will give (10 x 60000) so add bias to each neuron
 that is (10 x60000) + (10,1) (bias is basically to avoid inactivity and make the model independent even if input is zero) then z = (10 x 60000) then pass each 10 neuron to ReLU(z) 
 ( this makes the model adapt to curves instead of just straight lines) then for second hidden layer z2 = Wz +b now we know that z =(10 x 60000) so
 second hidden layer also has 10 neurons so each neuron must scan 10 previous neuron to find a curve which is (10 x10) so W.z gives
 (10 x10) x (10 x 60000) which is agin (10 x 60000) so add bias for this (10x 60000) + (10 ,1) which gives z2 = (10x 60000) now apply softmax for each
 to find probabilistic values which is between [0,1] and sum of all  p(x) values of output neurons = 1 softmax is only for output layer
 (Higher the value for a neuron more likely the prediction)'''

x_train = x_train.T 
x_test = x_test.T #For testing


#First initialize the parameters
def init_parameters():
    W1 = np.random.rand(10,784)*0.01 #Lower the weights , stable the neuron
    b1 = np.random.rand(10,1)
    W2 = np.random.rand(10,10) *0.01
    b2 = np.random.rand(10,1)

    return W1,b1,W2,b2

#Activation function
def ReLU(z):
    return np.maximum(0,z) #if x>0 then x if x<=0 then 0

#Forward pass
def forward_prop(X , W1 , b1 , W2 , b2):
    z1 = W1 @ X + b1
    a1 = ReLU(z1)

    #Second layer
    z2 = W2 @ a1 + b2
    predictions = softmax(z2)
    return a1 , z1 , predictions 

#Softmax function
def softmax(z): # goes like s = e^z(i) / sumof(e^z(j)) where j = z2(1),z2(2),z2(3),...,z2(10) and i is z2 current neuron value
    shifted_z = z - np.max(z ,axis=0) #note our z2 is (10 x 60000) so axis=0 operates downwards for each row 
    exp_z = np.exp(shifted_z) #we use e^x (euler number) because e^x is always >0 even for -ve of x (since softmax is prob no -ve value should be there)
    return exp_z/np.sum(exp_z,axis=0)


#Intialize the weights once 
W1 , b1 , W2 , b2 = init_parameters()
result = forward_prop (x_train ,W1 , b1 , W2,b2)

'''
Now comes the Loss function 
The loss function we will use is Cross Entropy which gives us the exact loss of our model from the original data 
  Mathematically it is expressed as 
      L = - sum( y(i) log((y^(i))))'''


def one_hot_encode(y , num_classes=10):
    return np.eye(num_classes)[y].T #Eye creates a zero matrix  with diagnals as 1 (num_classes =10 because there are numbers from 0-9)

def cross_entropy(predictions ,y_true):
    epsilon = 1e-12 #if prediction =0 then log(0) is undefined so we add epsilon 1e-12 which means 1 x 10^(-12) = 0.0000..1

    entropy = - np.sum(y_true * np.log(predictions +epsilon) ,axis =0)

    return np.mean(entropy) #because there will be 60000 entropy values we need one for accuracy so sum/60000

'''Now comes the BACKPROPOGATION , it is way that makes network to learn its mistake by traversing back and adjusts weights and biases
how it is adjusted ? It is done using a method called gradient descent

W(new) = W(old) - (alpha) (pd of weights) '''

def backward_prop(X, a1, z1, predictions, y_true, W2, m):
    #For second hidden layer

    #calculating the standard error dz2
    dz2 = predictions - y_true

    #Calculating weighted graidents dW2
    dW2 = (dz2 @ a1.T) /m

    #calculating bias gradient
    db2 = np.sum(dz2 ,axis=1 , keepdims=True)/m   #This keepdims changes (m,) to (m,1) so that at future it wont create mismatch dimensions error

    '''Intuitions : 1.dz2 says how much preactivation(z2) needs to change to reduce the loss
                    2.dW2 says how much we need to nudge a specific weight in W2 to reduce the loss
                    3.db2 says should I need to rise or lower the thershold of this specific neuron'''
    #For first hidden layer

    #Caculating the error for first layer dz1 
    dz1 = (W2.T @ dz2) * (z1 >0) # Z1 >0  for ReLU derivative if z1>0 returns 1 if z1<=0 returns 0
    #At first layer error chain rule comes in so that everything is accumulated

    #Calculating weighted gradient dW1
    dW1 = (dz1 @ X.T)/m

    #Calculating bias gradient
    db1 = np.sum(dz1 , axis =1 , keepdims=True)/m

    return dW1, db1, dW2, db2


 #Now comes the updation part using gradient descent 
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - (alpha * dW1)
    b1 = b1 - (alpha * db1)
    W2 = W2 - (alpha * dW2)
    b2 = b2 - (alpha * db2)

    return W1 , b1 , W2 , b2

loss_history = [] #For graph    
#Now train the network
def train(x_train, y_train, iterations, alpha):
    # Initialize Parameters
    W1, b1, W2, b2 = init_parameters()
    y_train_one_hot = one_hot_encode(y_train)
    m = x_train.shape[1]


    print("Starting training...") #Just for indication

    for i in range(iterations):
        #Forward Pass
        a1, z1, predictions = forward_prop(x_train, W1, b1, W2, b2)

        #Backpropagation
        dW1, db1, dW2, db2 = backward_prop(x_train, a1, z1, predictions, y_train_one_hot, W2, m)

        #Update Parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        #Monitor Loss
        if i % 10 == 0:
            loss = cross_entropy(predictions, y_train_one_hot)
            loss_history.append(loss) #For graph
            print(f"Iteration {i}: Loss = {loss:.4f}")

    print("Training complete!") #Just for notation
    return W1, b1, W2, b2

# Set your learning rate and iterations
learning_rate = 0.1
epochs = 500

W1_final, b1_final, W2_final, b2_final = train(x_train, y_train, epochs, learning_rate)

#Plotting it for reference
plt.plot(loss_history)
plt.title("Model Learning Progress")
plt.xlabel("Iteration (x10)")
plt.ylabel("Cross-Entropy Loss")
plt.show()


#Now training with testing sets
def get_predictions(probabilities):
    # np.argmax returns the index of the highest probability
    return np.argmax(probabilities, axis=0)

def get_accuracy(predictions, y_true):
    # Calculate how many match
    return np.sum(predictions == y_true) / y_true.size



a1_test, z1_test, test_probs = forward_prop(x_test, W1_final, b1_final, W2_final, b2_final)
test_predictions = get_predictions(test_probs)

accuracy = get_accuracy(test_predictions, y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")


















# In[ ]:




