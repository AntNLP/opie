# Introduction

- The test set in paper "Large-scale Opinion Relation Extraction with Distantly Supervised Neural " [2017 EACL].
 
- Reviews are from [Amazon datasets](http://jmcauley.ucsd.edu/data/amazon/) (4 domains: Phone, Movie, Food and Pet).

- The annotation tool is [brat](http://brat.nlplab.org/), and the data follows brat's [format](http://brat.nlplab.org/manual.html).


# Format

Both the raw text (sentence split and word tokenized) and annotations are provied for each review:

- review_%d.txt: a raw review text. Each row is a sentence.

- review_%d.ann: an annotation file for review_%d.txt:

  - Each row is an annotation, which could be an opinion target, an opinion expression or a relation.

  - An opinion target row (opinion expression is similar) contains three fields (separated by tab): `T%d``\t``OpinionTarget left right``\t``RawString`.
  	- `T%d`: id of the target.
  	- `OpinionTarget left right`: offsets in the plain text.
  	- `RawString`: raw string of the opinion target.

  - A relation row contains two fields:
  	- `R%d`: id of the relation.
  	- `Arg1:T%d`: id of the opinion expression.
  	- `Arg2:T%d`: id of the opinion target.
