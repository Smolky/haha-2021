# UMUTeam at HAHA 2021
## Linguistic Features and Transformers for Analysing Spanish Humor. The What, the How, and to Whom
Giving computers basic notions of humour can result in better and more emphatic user interfaces that are perceived more natural. Understanding humor, however, is challenging because it is subjective as well as cultural and background dependant. Another difficulty is that sharpen forms of humor rely on figurative language, in which words loss their literally meaning, to produce word play or puns. These facts hinders Natural Language Processing tasks for handling texts in which some forms of humor sense are present. Moreover, it is worth noting that humor can be used as a Troy Horse to introduce oppressive-speech passing them off as harmless jokes. Therefore, humor detection has been proposed as shared-task in workshops in the last years. In this paper we describe our participation in the HAHA'2021 shared task, regarding fine-grain humor identification in Spanish. This task is divided into four subtasks concerning what it is funny, if there is agreement about what it is funny, which mechanisms are used to do humor, and what topics are the jokes about. Our proposals to solve these subtasks are grounded on the combination of linguistic features and state-of-the-art transformers by means of neural networks. We achieved the 1st position for humor rating Funniness Score Prediction with a RMSE of 0.6226, the 8th position for humor classification subtask with an 85.44 F1-score of humours category, and the 7th and the 3rd position for the subtasks of humor mechanism and target classification with an macro averaged F1-score of 20.31 and 32.25 respectively. 


# Official leader board
| User         | Team          | Task 1 Score | Task 2 Score | Task 3 Score | Task 4 Score |
|--------------|---------------|--------------|--------------|--------------|--------------|
| TanishqGoel  | Jocoso        | 0.8850 (1)   | 0.6296 (3)   | 0.2916 (2)   | 0.3578 (2)   |
| icc          |               | 0.8716 (2)   | 0.6853 (9)   | 0.2522 (3)   | 0.3110 (4)   |
| kuiyongyi    |               | 0.8700 (3)   | 0.6797 (8)   | 0.2187 (5)   | 0.2836 (6)   |
| moradnejad   | ColBERT       | 0.8696 (4)   | 0.6246 (2)   | 0.2060 (7)   | 0.3099 (5)   |
| jgcarrasco   | noda risa     | 0.8654 (5)   | -            | -            | -            |
| Neakail      | BERT4EVER     | 0.8645 (6)   | 0.6587 (4)   | 0.3396 (1)   | 0.4228 (1)   |
| Mjason       |               | 0.8583 (7)   | 1.1975 (11)  | -            | -            |
| JAGD         | UMUTeam       | 0.8544 (8)   | 0.6226 (1)   | 0.2087 (6)   | 0.3225 (3)   |
| skblaz       |               | 0.8156 (9)   | 0.6668 (6)   | 0.2355 (4)   | 0.2295 (7)   |
| sgp55        | humBERTor     | 0.8115 (10)  | -            | -            | -            |
| antoniorv6   | RoBERToCarlos | 0.7961 (11)  | 0.8602 (10)  | 0.0128 (10)  | 0.0000 (9)   |
| lunna        |               | 0.7693 (12)  | -            | 0.0404 (9)   | -            |
| sarasmadi    | N&&N          | 0.7693 (12)  | -            | 0.0404 (9)   | -            |
| ayushnanda14 |               | 0.7679 (13)  | 0.6639 (5)   | -            | -            |
| noorennab    | Noor          | 0.7603 (14)  | -            | 0.0404 (9)   | -            |
| kdehumor     | KdeHumor      | 0.7441 (15)  | 1.5164 (12)  | -            | -            |
| baseline     | baseline      | 0.6619 (16)  | 0.6704 (7)   | 0.1001 (8)   | 0.0527 (8)   |


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
