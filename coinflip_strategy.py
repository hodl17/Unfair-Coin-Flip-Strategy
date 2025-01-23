# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:32:44 2025

@author: devli
"""
from random import choices
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import pandas as pd

class Coin:
    def __init__(self):
        self.outcomes = ['H', 'T']
        self.probabilities = [0.6, 0.4]
    
    def toss(self):
        return choices(self.outcomes, self.probabilities)[0]
    
class Experiment:
    def __init__(self, trials, stake=1.0):
        self.trials = trials
        self.stake = stake
        self.coin = Coin()
        self.capital=100
        self.capital_tracker = [self.capital]
        
    def run(self):
        for n in range(self.trials):
            toss = self.coin.toss()
            if toss == 'H':
                winnings=self.stake*self.capital*2
            else:
                winnings=0
            self.capital-=self.capital*self.stake
            self.capital+=winnings
            
            if self.capital==0:
                self.capital_tracker.append(self.capital)
                break
            
            self.capital_tracker.append(self.capital)
            
            
def run_simulation(n, trials, stakes):
        
    results_by_stake = {}
    compound_rates_by_stake = {}
    
    for s in stakes:
        return_paths = []
        for i in range(n):
            exp = Experiment(trials=trials, stake=s)
            exp.run()
            return_paths.append(exp.capital_tracker)
        
        returns = [(x[-1] - x[0])/x[0] for x in return_paths]
        compound_returns = [pow(x[-1]/x[0], 1/trials)-1 for x in return_paths]
        compound_rates_by_stake[s] = compound_returns
        
        mean_return = round(statistics.mean(returns),2)
        std = statistics.stdev(returns)
        sharpe = mean_return / std if std > 0 else mean_return
        
        negative_returns = [r for r in returns if r <= 0]
        negative_std = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0
        sortino = mean_return / negative_std if negative_std > 0 else mean_return
        
        risk_of_ruin = len(negative_returns) / len(returns)
        median_return = statistics.median(returns)
        
        # compounded rates are scaled better for graph comparisons
        mean_return_compounded = pow(mean_return/100, 1/trials)-1 if mean_return > 0 else 0
        median_return_compounded = pow(median_return/100, 1/trials)-1 if median_return > 0 else 0
        
        results_by_stake[s] = [mean_return, median_return, sharpe, sortino, risk_of_ruin, 
                               mean_return_compounded, median_return_compounded,
                               statistics.mean(compound_returns), statistics.median(compound_returns)]
    
    
    df = pd.DataFrame.from_dict(results_by_stake, orient='index', columns=['mean', 'median', 'sharpe', 'sortino', 'risk_of_ruin',  
                                                                           'mean_compounded', 'median_compounded', 
                                                                           'compound_mean', 'compound_median'])
    return df, compound_rates_by_stake

def plot_return_distribution(returns, stakes=[0.1, 0.2, 0.3]):
    for s in stakes:
        sns.kdeplot(returns[s], label=s)
    plt.legend()
    plt.title('Distribution of compound returns by percentage stake')
    plt.xlabel('Compound return')
    plt.show()
    
def plot_ror(returns):
    sns.lineplot(data=results_df.risk_of_ruin)
    plt.title('Risk of ruin')
    plt.ylabel('Probability')
    plt.xlabel('Stake')
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.show()
    
def barplot(plot_data, column, positive_only=True):
    if positive_only:
        plot_data = plot_data[plot_data[column]>0]
    sns.barplot(data=plot_data, x=plot_data.index, y=column)
    plt.xlabel('Stake')
    plt.show()

N = 10000 # how many monte carlo iterations to do
TRIALS = 1000 # max coin flips on each iteration
STAKES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # test a strategy that allocates this % of capital to each coin flip

results_df, compound_return_dict = run_simulation(N, TRIALS, STAKES)


    
    