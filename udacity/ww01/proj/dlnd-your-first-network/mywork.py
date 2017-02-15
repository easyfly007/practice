import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Bike-Sharing-Dataset/hour.csv'
riders = pd.read_csv(data_path)
print(type(riders))
