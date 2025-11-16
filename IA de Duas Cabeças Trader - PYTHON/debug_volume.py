from cherry import load_optimized_data_original
import numpy as np

df = load_optimized_data_original()
print(f"volume_1m stats:")
print(f"  Min: {df['volume_1m'].min()}")
print(f"  Max: {df['volume_1m'].max()}")
print(f"  Mean: {df['volume_1m'].mean()}")
print(f"  Std: {df['volume_1m'].std()}")
print(f"  Zeros: {np.sum(df['volume_1m'] == 0)}")
print(f"  Non-zeros: {np.sum(df['volume_1m'] > 0)}")
