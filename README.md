# chula-course-recommender-demo
Course recommender system by content based filtering method demo application. Based on this [Proof of concept repo](https://github.com/new5558/Chula-course-recommender-proof-of-concept).

## Run project on localhost
`docker-compose up`

## [Demo website](https://share.streamlit.io/new5558/chula-course-recommender-demo/src/main.py)

![image](https://user-images.githubusercontent.com/12471844/149357870-79eae41a-122e-45fa-a533-5962aed63991.png)

## Method
I trained [sentences-transformers](https://www.sbert.net/) to create courses pair embedding vectors then calculate simialrity from vectors. The embedding model used course description as its input sentences.    
![image](https://user-images.githubusercontent.com/12471844/149375295-ab65c387-7ae6-49ee-a7e5-df916156e04b.png)
### Data, Training, and Similarity calculation
I used courses description and study program provided publicly from [Academic Chula website](http://www.academic.chula.ac.th/search/) The following rules are used to inferred the similarity between courses pair.

1. Courses pair that are in the same study program has high simialrity. The more study program they have in common, the more simialrity it is.
2. Courses that are in same preffered year of study and same study program will have extra similarity score. According the Acadmuc Chula, The fifth number `N` of course number or `XXXXNX` can be inferred as preferred year of study.  

I combined and normalized similarity scores both rules to create similarity matrix. To avoid class imbalance problem, I downsampled courses pair that has zero similarity scores to matched with the pairs that have possitive similarity score.

## Evaluation
I evaluated course embedding result using faculty prediction task. the model were given course description and has to provide course embedding vector. the embedding are inputted into basic KNN model and achieved around 85% prediction accuracy on validation set. The result can be found on [Proof of concept repo](https://github.com/new5558/Chula-course-recommender-proof-of-concept).
