# UMUTeam at HAHA 2021
## Linguistic Features and Transformers for Analysing Spanish Humor. The What, the How, and to Whom
Giving computers basic notions of humour can result in better and more emphatic user interfaces that are perceived more natural. Understanding humor, however, is challenging because it is subjective as well as cultural and background dependant. Another difficulty is that sharpen forms of humor rely on figurative language, in which words loss their literally meaning, to produce word play or puns. These facts hinders Natural Language Processing tasks for handling texts in which some forms of humor sense are present. Moreover, it is worth noting that humor can be used as a Troy Horse to introduce oppressive-speech passing them off as harmless jokes. Therefore, humor detection has been proposed as shared-task in workshops in the last years. In this paper we describe our participation in the HAHA'2021 shared task, regarding fine-grain humor identification in Spanish. This task is divided into four subtasks concerning what it is funny, if there is agreement about what it is funny, which mechanisms are used to do humor, and what topics are the jokes about. Our proposals to solve these subtasks are grounded on the combination of linguistic features and state-of-the-art transformers by means of neural networks. We achieved the 1st position for humor rating Funniness Score Prediction with a RMSE of 0.6226, the 8th position for humor classification subtask with an 85.44 F1-score of humours category, and the 7th and the 3rd position for the subtasks of humor mechanism and target classification with an macro averaged F1-score of 20.31 and 32.25 respectively. 


## Notes
Due to space limitation in GitHub, the features are not included. For the linguistic features, you can request them to joseantonio.garcia8@um.es


## Install
1. Create a virtual environment in Python 3
2. Install the dependencies that are stored at requirements.txt
3. Create the folder 
    ```assets/haha/2021-es```
    
4. Generate the dataset: 
    ```python -W ignore compile.py --dataset=haha --corpus=2021-es```
    
5. Create symbolink links for the dataset in each task in folders:
    ```assets/haha/2021-es/task-1```
    ```assets/haha/2021-es/task-2```
    ```assets/haha/2021-es/task-3```
    ```assets/haha/2021-es/task-4```
    
6. Finetune BERT. 
    ```python -W ignore train.py --dataset=haha --corpus=2021-es --model=transformers```
    
7. Feature selection for each task. 
    ```python -W ignore feature-selection.py --dataset=haha --corpus=2021-es --task=task-1```
    ```python -W ignore feature-selection.py --dataset=haha --corpus=2021-es --task=task-2```
    ```python -W ignore feature-selection.py --dataset=haha --corpus=2021-es --task=task-3```
    ```python -W ignore feature-selection.py --dataset=haha --corpus=2021-es --task=task-4```
    
8. Generate BF features: 
    ```python -W ignore generate-bf.py --dataset=haha --corpus=2021-es --task=task-1```
    ```python -W ignore generate-bf.py --dataset=haha --corpus=2021-es --task=task-2```
    ```python -W ignore generate-bf.py --dataset=haha --corpus=2021-es --task=task-3```
    ```python -W ignore generate-bf.py --dataset=haha --corpus=2021-es --task=task-4```
    
9. Feature selection again for the BF features. 
    ```python -W ignore feature-selection.py --dataset=haha --corpus=2021-es --task=task-1```
    ```python -W ignore feature-selection.py --dataset=haha --corpus=2021-es --task=task-2```
    ```python -W ignore feature-selection.py --dataset=haha --corpus=2021-es --task=task-3```
    ```python -W ignore feature-selection.py --dataset=haha --corpus=2021-es --task=task-4```
    
10. Train. 
    ```python -W ignore train.py --dataset=haha --corpus=2021-es --model=deep-learning --features=lf --task=task-1```
    ```python -W ignore train.py --dataset=haha --corpus=2021-es --model=deep-learning --features=lf --task=task-2```
    ```python -W ignore train.py --dataset=haha --corpus=2021-es --model=deep-learning --features=lf --task=task-3```
    ```python -W ignore train.py --dataset=haha --corpus=2021-es --model=deep-learning --features=lf --task=task-4```
    
11. Evaluate. 
    ```python -W ignore haha-submission.py --dataset=haha --corpus=2021-es```
