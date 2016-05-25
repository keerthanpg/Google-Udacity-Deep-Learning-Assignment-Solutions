"""Softmax."""
import numpy as np
import math

scores = np.asarray([1.0, 2.0, 3.0])
'''scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])'''



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
        
    if type(x)is list:# if object passed is a list
        softmaxx=[]
        sum_i=0
        for element in x:
            sum_i+=math.exp(element)
        
        for element in x:
            softmaxx.append(math.exp(element)/sum_i)
    else:# if object passed is a 2-d array
        dimension=x.shape
        if len(dimension)>1:         
        
            softmaxx=[]
            for i in xrange( dimension[0]):
                column=x[i, :]            
                softmax_sample=[]
                sum_i=float(0)
                for element in column:
                    sum_i+=math.exp(element)

                for element in column:
                    softmax_sample.append(math.exp(element)/sum_i)

                softmaxx.append(softmax_sample)
            softmaxx=np.asarray(softmaxx)
            #softmaxx=np.transpose(softmaxx)

        else:
            softmaxx=[]
            sum_i=float(0)
            for element in x:
                sum_i+=math.exp(element)
        
            for element in x:
                softmaxx.append(math.exp(element)/sum_i)
      
    return softmaxx

print (softmax(scores))


# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
softmax(scores)
plt.plot(x, np.transpose(softmax(scores)), linewidth=2)
plt.show()
