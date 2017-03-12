===========================================================

# Introduction

- This README for the [Amazon datasets](http://jmcauley.ucsd.edu/data/amazon/)(only test, including 4 domains: Phone, Movie, Food, Pet).

- The annotation tool we use was brat rapid [annotation tool](http://brat.nlplab.org/), and the brat rapid annotation tool manual is [here](http://brat.nlplab.org/manual.html).

- This data belong to the "Large-scale Opinion Relation Extraction with Distantly Supervised Neural Network
" which was accepted by [2017 EACL](http://www.eacl2017.org/index.php/program/accepted-papers).

===========================================================

# Data Format Summary

The review text(after word_tokenize and sent_tokenize) and annotation file are under each domain file(Food, Movie, Pet, Phone).

- Each review_%d.txt file has a raw review text. Each row represents a sentence.

- Each review_%d.ann file is annotation file. It follows brat annotation format. The formats are explained in the following.

  - Each row represents a annotation with opinion target or opinion expression or relation

  - For opinion target row, it like `T%d``\t``OpinionTarget left right``\t``ExactString`.
  	- `T%d`: The `%d`th row is the opinion target row.
  	- `OpinionTarget left right`: The offsets in the plain text representation of the Amazon review are denoted by `left` and `right`.
  	- `ExactString`: The exact string representation of the opinion target.

  - For opinion expression row, it like `T%d``\t``OpinionExpression left right``\t``ExactString`.
  	- `T%d`: The `%d`th row is the opinion expression row.
  	- `OpinionExpression left right`: The offsets in the plain text representation of the Amazon review are denoted by `left` and `right`.
  	- `ExactString`: The exact string representation of the opinion expression.

  - For relation row, it like `R%d``\t``ExpressionTarget Arg1:T%d Arg2:T%d`.
  	- `R%d`: The `%d`th row is the relation row.
  	- `Arg1:T%d`: The opinion expression.
  	- `Arg2:T%d`: The opinion target.
