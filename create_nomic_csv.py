# goal: generation_id | prompt_id | prompt | embedding

import csv 

import numpy as np

a = np.asarray([[1,2,3],[13,32,32]])

with open("test.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["generation_id", "prompt_id", "prompt", "embeddings"])
    writer.writerow(["dsn-sndfjnps-sdjnfpsd-dnf", "dfnd-fsfds-fs-fdsf-jnk", "cute cat of course a cute cat", a.tostring()])

with open("test.csv", "r") as f:
    reader = csv.reader(f)
    headers = next(reader)
    row = next(reader)

    print(np.fromstring(row[-1], dtype=int))



