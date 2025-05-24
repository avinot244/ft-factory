from tqdm import tqdm
import time

my_list : list[tuple[int, int]] = [(1, 2), (3, 4), (5, 6)]

for (index, (a, b)) in tqdm(enumerate(my_list), total=len(my_list)):
    time.sleep(0.5)  # Simulating some work
    tqdm.write(f"Index: {index}, a: {a}, b: {b}")