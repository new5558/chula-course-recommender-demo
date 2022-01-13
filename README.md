# chula-course-recommender-demo
Course recommender system by content based filtering method demo application. This project is a part of Data team experiments at www.cugetreg.com

## Run project on localhost
`docker-compose up`

## [Demo website](https://share.streamlit.io/new5558/chula-course-recommender-demo/src/main.py)

![image](https://user-images.githubusercontent.com/12471844/149357870-79eae41a-122e-45fa-a533-5962aed63991.png)

## Notion
We want to create a recommendation system that do not rely on users data. The current system CU Get Reg is using depends on users add course history and subject to Cold start problem. We also aware of PDPA laws that will be enacted in this June 2022 and affect our current recommendation system's performance. 

## Method
The model used/trained in this repo is [sentences-transformers](https://www.sbert.net/). Sentences transformers can create courses embedding vectors that can be used to calculate courses pair simialrity. The embedding model used course description as its input sentences.    
![image](https://user-images.githubusercontent.com/12471844/149375295-ab65c387-7ae6-49ee-a7e5-df916156e04b.png)
### Data, Training, and Similarity calculation
The main dataset on the repo are courses description and study program, provided publicly from [Academic Chula website](http://www.academic.chula.ac.th/search/) The following rules are used to inferred the similarity between courses pair.

1. Courses pair that are in the same study program has high simialrity. The more study program they have in common, the more simialrity it is.
2. Courses that are in same prefered year of study and same study program will have extra similarity score. According acadmic Chula, The fifth number `N` of course number(`XXXXNX`) can be inferred as prefered year of study.  

The similarity scores of both rules are combined and normalized to create 20000 X 20000 courses similarity matrix. The matrix has to be flatten to use as Sentence transformers training labels. To reduce training time and avoid class imbalance problem, courses pair that has zero similarity scores was downsampled to matched with the pairs that have possitive similarity score. The training code is on this [Proof of concept repo](https://github.com/new5558/Chula-course-recommender-proof-of-concept).

## Evaluation
The course embedding model is evaluated by using faculty prediction task. the model were given course description and has to provide course embedding vector. the embedding are inputted into basic KNN model and achieved around 85% prediction accuracy on validation set. The full result can be found on [Proof of concept repo](https://github.com/new5558/Chula-course-recommender-proof-of-concept).

## Possible Improvements
- In reality, the prefered year listed in courses number can slightly deviate from actual year student have to register by one or two. This is because curriculum revised and some class are likned with multiple study programs. With this knowledge, we may able to revise the course simialirty calculation rules by scaling preffered year difference as simiality score. For example, course with same year and study program will have highest simialirty score while score with small year difference but same study program will have slightly lower similarity score in order.
- Another way to improve infered simialrity score is to model the relationship between study programs. This is based on the notion that people taken similar program should interest in similar courses.
- The current model is trained on 90% or all courses data due to train test split rules. We should train model with all courses.
- The downsampling technique used in training is scalable in large samples. We can intuitively select negative sample for training using various Computer Vision techniques ex. [Online and Offline tripet mining](https://omoindrot.github.io/triplet-loss).
