import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calcMSE(values, optimal_values):
    result = []
    for state in optimal_values.keys():
        if state in [(4,4), (2,2), (3,2)]:
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

def plotCSV(csv_file, optV, title):
    groupedData = csvToDict(csv_file)
    mseVals = []
    outerKeys = sorted(groupedData.keys())
    for outerKey in outerKeys:
        innerKey = groupedData[outerKey]
        mse = calcMSE(innerKey, optV)
        mseVals.append(mse)
    plt.figure(figsize=(10, 6))
    plt.plot(outerKeys, mseVals, color="blue", marker="o", linestyle="-", label="MSE")
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()

optV = {
    (0, 0): 3.9148,
    (0, 1): 3.7265,
    (0, 2): 3.4056,
    (0, 3): 5.7106,
    (0, 4): 6.4474,
    (1, 0): 4.4703,
    (1, 1): 4.9468,
    (1, 2): 5.1033,
    (1, 3): 6.6127,
    (1, 4): 7.3889,
    (2, 0): 5.0614,
    (2, 1): 5.6538,
    (2, 3): 7.5769,
    (2, 4): 8.4637,
    (3, 0): 5.7274,
    (3, 1): 6.4761,
    (3, 3): 8.5738,
    (3, 4): 9.6946,
    (4, 0): 6.4761,
    (4, 1): 7.4223,
    (4, 2): 8.5123,
    (4, 3): 9.6946,
    (4, 4): 0.0000
}

csv_files = ["./MCTS/GridWorld687Results/regularMCTSExploreTwo.csv", "./MCTS/GridWorld687Results/regularMCTSExploreSqrtTwo.csv", "./MCTS/GridWorld687Results/regularMCTSExploreOne.csv", "./MCTS/GridWorld687Results/regularMCTSExploreOneOverSqrtTwo.csv", "./MCTS/GridWorld687Results/regularMCTSExplorePointZeroFive.csv", "./MCTS/GridWorld687Results/regularMCTSExplorePointZeroZeroOne.csv", "./MCTS/GridWorld687Results/EpsilonPointTwoMCTSExplorePointZeroZeroOne.csv", "./MCTS/GridWorld687Results/EpsilonPointOneMCTSExplorePointZeroZeroOne.csv","./MCTS/GridWorld687Results/EpsilonPointZeroFiveMCTSExplorePointZeroZeroOne.csv", "./MCTS/GridWorld687Results/EpsilonDecayMCTSExplorePointZeroZeroOne.csv"]
titles = ["regular MCTS Exploration=2 MSE", "regular MCTS Exploration=sqrt(2) MSE", "regular MCTS Exploration=1 MSE", "regular MCTS Exploration=1/sqrt(2) MSE", "regular MCTS Exploration=0.05 MSE", "regular MCTS Exploration=0.001 MSE", "Epsilon=0.2 MCTS Exploration=0.001 MSE", "Epsilon=0.1 MCTS Exploration=0.001 MSE", "Epsilon=0.05 MCTS Exploration=0.001 MSE", "Epsilon decay by 0.02375 every 250 episodes MCTS Exploration=0.001 MSE"]

for csv_file, title in zip(csv_files, titles):
    plotCSV(csv_file, optV, title)

