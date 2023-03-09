# ECG classification for microcontroller

<img width="400" alt="notebook-visual" src="https://user-images.githubusercontent.com/54487578/224147267-a733eda9-c9de-4797-a45d-842b0f3c5632.png">

# Instructions

### 1. Clone repository & cd into directory
```
git clone https://github.com/antoniog11/embedded-ecg-classifier.git
```

```
cd embedded-ecg-classifier
```

### 2. Create a virtual environment
```
python3 -m venv .ecg-classif
```

### 3. Activate the environment
```
source .ecg-classif/bin/activate
```

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Download dataset, unzip, and put into embedded-ecg-classifier (main repo) folder
Download link [here](https://drive.google.com/file/d/1zSOZ-mi6fjuZ9A6QzxPqJap5hQRhGkdI/view?usp=sharing)

### 6. Run notebook
```
jupyter notebook
```


### Troubleshooting
1. You may need to re-run the `pip install wfdb` command inside jupyter notebook if the wfdb module is not found
2. If the data isn't populating, check that you've (1) downloaded the dataset and (2) put it in the main directory 
