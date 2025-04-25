# How to run our art classifier

1) Download data from wikiart into same root folder as code. Download from this GitHub repository using the first https URL (https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset). Also download the genre_class.txt, genre_train.csv, and genre_val.csv here (https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset/Genre).

2) Make sure the folder structure looks like the following
- root folder
  - data
    - wikiart
      - Abstract_Expressionism
      - Action_painting
      - Analytical_Cubism
      - ... etc
    - genre_class.txt
    - genre_train.csv
    - genre_val.csv
  - art-classifier-weights.pt
  - art_loader.py
  - art_resnet.py
  - main.py
  - trainer.py

3) Run code for model training or inference by typing python main.py in command line. If running training, comment out inference code. If running inference, comment out training code.