# Exam project for Natural Language Processing

## About the project
This is the exam project of:
- Nanna Marie Steenholdt (NMS), 201805892@post.au.dk
- Julia Flora Fürjes (JFF), 202006018@post.au.dk
- Bianka Szöllösi (BIS), biankasz@mgmt.au.dk

> [!IMPORTANT]
> The dataset is not available here due to NDA restrictions.

## Setting up the project

In the repository, we have included the best performing model, called `random_forest.py`. Here is how you can run it:

1. Acquire the dataset and save it as `job_ads_updated.xlsx`.

2. Make sure that you are in the right directory.

3. Create a virtual environment by running the following code in the terminal:

```
sudo apt-get update
sudo apt-get install python3-venv
python -m venv path_to_folder/venvs
```

4. Activate the virtual environment by running the following code in the terminal:

```
# Only run the first two lines if you have not done it yet in this run
sudo apt-get update
sudo apt-get install python3-venv
source venvs/bin/activate
pip install -r requirements.txt
```

5. Select interpreter (`venvs/bin/python`).

6. Run the code.
