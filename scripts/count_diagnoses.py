import pandas as pd

df = pd.read_csv("../data/example_ground_truth.csv")

count_healthy = 0
count_melanoma = 0
count_seborrheic_keratosis = 0

for i in df:
    if df[i,1] == 1.0:
        count_melanoma +=1
    elif df[i,2] == 1.0:
        count_seborrheic_keratosis +=1
    else:
        count_healthy +=1

print(count_healthy, count_melanoma, count_seborrheic_keratosis)