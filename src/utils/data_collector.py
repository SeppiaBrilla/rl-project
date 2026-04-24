from abc import abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataCollector : 

    """
    A utility class for tracking and persisting reinforcement learning training data.
    
    This class uses static methods and class-level attributes to collect episode 
    rewards across different instances and save them to a structured CSV format.
    """
    episode_rewards_list = [] #Reward list, used to save keep a track of collected rewards
    
    @staticmethod 
    def appendata(episode_reward):
        """
        Appends the total reward of a single completed episode to the global tracker.
        Args:
            episode_reward (float/int): The cumulative reward earned in the episode.
        """
        DataCollector.episode_rewards_list.append(episode_reward)

    @staticmethod
    def createSave(args):
        """
        Converts the collected rewards into a Pandas DataFrame and exports to CSV.
        
        The filename is dynamically generated using the algorithm and environment 
        names provided in the args object.
        
        Args:
            args: An object (typically argparse.Namespace) containing 'algo' 
                  and 'env' attributes for naming the output file.
        """
        dataframe = pd.DataFrame({"episode": range(len(DataCollector.episode_rewards_list)), "reward": DataCollector.episode_rewards_list}) #saving data to dataFrame
        dataframe.to_csv(f"results_{args.algo}_{args.env}.csv", index=False)  #save to file

