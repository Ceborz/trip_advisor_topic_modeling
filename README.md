## Project setup
Install conda
Create enviroment for conda
In the enviroment run
```
conda install python
conda install -c anaconda pandas
conda install -c conda-forge matplotlib
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m spacy download es_core_news_lg
```
## To run training
Go inside the folder of the project (topic_modeling)
```
cd src
python -m data.prepare_data
python -m features.tokenize_data
python -m models.train
```
## To run test
The file that will be tested is data/test/test_prediction.txt. This file will run by line to predict a topic. So edit the file chaning to lines you want to predict the topic
After that run the prediction
Inside the src folder
```
python -m models.predict
```
This will print the result