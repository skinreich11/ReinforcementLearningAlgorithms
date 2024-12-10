import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mse(values, optimal_values):
    result = []
    for state in optimal_values.keys():
        if state in [(4,4), (2, 1), (2, 2), (2, 3), (3, 2)]:
            continue
        else:
            result.append((values[state][1]-optimal_values[state]) ** 2) 
    return np.mean(result)

def csvToDict(csv_file):
    data = pd.read_csv(csv_file)
    groupedData = {}
    for _, row in data.iterrows():
        outerKey = row['Outer Key']
        innerKey = eval(row['Inner Key (Tuple)'])
        value = eval(row['Value'])
        if outerKey not in groupedData:
            groupedData[outerKey] = {}
        groupedData[outerKey][innerKey] = value
    return groupedData

def plotCSV(csv_file, optimal_values, title):
    groupedData = csvToDict(csv_file)
    mseVals = []
    outerKeys = sorted(groupedData.keys())
    for outer_key in outerKeys:
        inner_key_values = groupedData[outer_key]
        mse = mse(inner_key_values, optimal_values)
        mseVals.append(mse)
    plt.figure(figsize=(10, 6))
    plt.plot(outer_key, mseVals, color="blue", marker="o", linestyle="-", label="MSE")
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()

optV = {
    (0, 0): 2.6638, (0, 1): 2.9969, (0, 2): 2.8117, (0, 3): 3.6671, (0, 4): 4.8497,
    (1, 0): 2.9713, (1, 1): 3.5101, (1, 2): 4.0819, (1, 3): 4.8497, (1, 4): 7.1648,
    (2, 0): 2.5936, (2, 1): 0.0,     (2, 2): 0.0,     (2, 3): 0.0,     (2, 4): 8.4687,
    (3, 0): 2.0992, (3, 1): 1.0849, (3, 2): 0.0,     (3, 3): 8.6097, (3, 4): 9.5269,
    (4, 0): 1.0849, (4, 1): 4.9465, (4, 2): 8.4687, (4, 3): 9.5269, (4, 4): 0.0
} 
csv_files = ["./MCTS/CatVSMonstersResults/regularMCTSExploreTwo.csv", "./MCTS/CatVSMonstersResults/regularMCTSExploreSqrtTwo.csv", "./MCTS/CatVSMonstersResults/regularMCTSExploreOne.csv", "./MCTS/CatVSMonstersResults/regularMCTSExploreOneOverSqrtTwo.csv", "./MCTS/CatVSMonstersResults/regularMCTSExplorePointZeroFive.csv", "./MCTS/CatVSMonstersResults/regularMCTSExplorePointZeroZeroOne.csv", "./MCTS/CatVSMonstersResults/EpsilonPointTwoMCTSExplorePointZeroZeroOne.csv", "./MCTS/CatVSMonstersResults/EpsilonPointOneMCTSExplorePointZeroZeroOne.csv","./MCTS/CatVSMonstersResults/EpsilonPointZeroFiveMCTSExplorePointZeroZeroOne.csv", "./MCTS/CatVSMonstersResults/EpsilonDecayMCTSExplorePointZeroZeroOne.csv"]
titles = ["regular MCTS Exploration=2 MSE", "regular MCTS Exploration=sqrt(2) MSE", "regular MCTS Exploration=1 MSE", "regular MCTS Exploration=1/sqrt(2) MSE", "regular MCTS Exploration=0.05 MSE", "regular MCTS Exploration=0.001 MSE", "Epsilon=0.2 MCTS Exploration=0.001 MSE", "Epsilon=0.1 MCTS Exploration=0.001 MSE", "Epsilon=0.05 MCTS Exploration=0.001 MSE", "Epsilon decay by 0.02375 every 250 episodes MCTS Exploration=0.001 MSE"]
for csv_file, title in zip(csv_files, titles):
    plotCSV(csv_file, optV, title)

