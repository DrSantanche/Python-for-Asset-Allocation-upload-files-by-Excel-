# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:06:03 2017

@author: marco
"""
import pandas
import numpy
filename='C:\Users\yourfile.csv'
raw_data=open(filename)
data=pandas.read_csv(raw_data,sep=';',header=0,index_col=0)
a=data.shape
b=a[0]
c=a[1]
Ret=(numpy.log(data)-numpy.log(data.shift(1)))*100
import matplotlib.pyplot as plt
plt.plot(Ret.values[0:,0])
plt.title('Asset1')
plt.plot(Ret.values[0:,1])
plt.title('Asset2')
plt.plot(Ret.values[0:,2])
plt.title('Asset3')
plt.plot(Ret.values[0:,3])
plt.title('Asset4')
plt.plot(Ret.values[0:,4])
plt.title('Asset5')
#etc.
#Asset allocation Markowitz
#Expected returns (Markowitz)
Exp_Ret=numpy.random.randn(1,c)
for i in range (0,c):
    Exp_Ret[:,i]=numpy.mean(Ret.values[1:,i])
Sigma=numpy.cov(Ret.values[1:,:],rowvar=False)
#Portfolios to generate
num_port=50000
#Storage for random weights
results = numpy.zeros((3+c,num_port))
#for loop to build portfolios
for i in xrange(num_port):
    weights=numpy.array(numpy.random.random(c))
    weights /=numpy.sum(weights)
    #sum of weights constrained to 1
        #calculate portfolio return and volatility
    portfolio_return = numpy.sum(Exp_Ret * weights)
    portfolio_std_dev = numpy.sqrt(numpy.dot(weights.T,numpy.dot(Sigma, weights)))
    #store results in results array
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2,i] = results[0,i] / results[1,i]
    #iterate through the weight vector and add data to results array
    for j in range(len(weights)):
        results[j+3,i] = weights[j]
 
#convert results array to Pandas DataFrame
results_frame = pandas.DataFrame(results.T,columns=['ret','stdev','sharpe','Asset1','continue typing asset names'])
 
#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
#locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
 
#create scatter plot coloured by Sharpe Ratio
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()
#plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=1000)
#plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=1000)
    