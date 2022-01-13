# chula-course-recommender-demo
Course recommender system by content based filtering method demo application.

## Run project on localhost
`docker-compose up`

## [Demo website](https://share.streamlit.io/new5558/chula-course-recommender-demo/src/main.py)

![image](https://user-images.githubusercontent.com/12471844/149357870-79eae41a-122e-45fa-a533-5962aed63991.png)

## Method
The model used/trained in this repo is [sentences-transformers](https://www.sbert.net/). Sentences transformers can create courses embedding vectors that can be used to calculate courses pair simialrity. The embedding model used course description as its input sentences.    
![image](https://user-images.githubusercontent.com/12471844/149375295-ab65c387-7ae6-49ee-a7e5-df916156e04b.png)
### Data, Training, and Similarity calculation
The main dataset on the repo are courses description and study program, provided publicly from [Academic Chula website](http://www.academic.chula.ac.th/search/) The following rules are used to inferred the similarity between courses pair.

1. Courses pair that are in the same study program has high simialrity. The more study program they have in common, the more simialrity it is.
2. Courses that are in same prefered year of study and same study program will have extra similarity score. According acadmic Chula, The fifth number `N` of course number(`XXXXNX`) can be inferred as prefered year of study.  

The similarity scores of both rules are combined and normalized to create 20000 X 20000 courses similarity matrix. To reduce training time and avoid class imbalance problem, courses pair that has zero similarity scores was downsampled to matched with the pairs that have possitive similarity score. The training code is on this [Proof of concept repo](https://github.com/new5558/Chula-course-recommender-proof-of-concept).

## Evaluation
The course embedding model is evaluated by using faculty prediction task. the model were given course description and has to provide course embedding vector. the embedding are inputted into basic KNN model and achieved around 85% prediction accuracy on validation set. The full result can be found on [Proof of concept repo](https://github.com/new5558/Chula-course-recommender-proof-of-concept).
