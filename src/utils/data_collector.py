from abc import abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataCollector : 
    """
        This class represents different 
    """
    episode_rewards_list = [] #Reward list, used to save keep a track of collected rewards
    
    @staticmethod 
    def appendata(episode_reward):
        DataCollector.episode_rewards_list.append(episode_reward)

    @staticmethod
    def createSave(args):
        dataframe = pd.DataFrame({"episode": range(len(DataCollector.episode_rewards_list)), "reward": DataCollector.episode_rewards_list}) #saving data to dataFrame
        dataframe.to_csv(f"results_{args.algo}_{args.env}.csv", index=False)  #save to file

