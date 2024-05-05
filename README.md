# vector-assignments-sgd

In this repo, we do not tackle any open problems in Mechanistic interpretability. I instead would like to propose a set of questions that I have been thinking about for the last few months inspired by research elsewhere. We hope that these questions seem well motivated.

Our aim is to study the geometry of vector embeddings. In particular, we want to study the embeddings of a transformer model trained for the purpose of sentence similarity/search using contrastive loss. We want to study this problem for the following reasons, 

1. Geometric intuition for superposition and its relation to isotropy of vector embeddings 
2. Optimality of gradient descent based transformer architectures
3. Demystify anisotropy of vector embeddings

Throughout this notebook we assume that the vectors are unit norm and that they are trained using some kind of contrastive loss. Unit norm forces the model to represent every feature in the output space, i.e. it cannot cop out by turning things off. While contrastive loss for search, allows us to interpret the geometry. As opposed to MLM or Auto-regressive loss where the geometry of the final layer is not necessarily constrained by any geometry; search transformers with contrastive loss literally demands similar vectors to be closer to each other and vice versa. Thus the objective of contrastive loss for search transformer allows us to study the geometry of the model output without us running into circles.



