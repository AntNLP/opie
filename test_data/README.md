===========================================================

# Introduction

This README (Janurary, 2017) for the Amazon datasets(only test, including 4 domains: Phone, Movie, Food, Pet)

The annotation tool we use was brat rapid annotation tool(http://brat.nlplab.org/), and the brat rapid annotation tool manual is here(http://brat.nlplab.org/manual.html)

===========================================================

# File overview

The foloder structure is as follows:

test_data

├── README.md

├── data

       ├── Food

       ├── Movie

       ├── Pet
 
       ├── Phone

===========================================================

# Data Format Summary

The review text(after word_tokenize and sent_tokenize) and annotation file are under each domain file(Food, Movie, Pet, Phone).

- Each review_%d.txt file has a raw review text. Each row represents a sentence.

- Each review_%d.txt file is annotation file. It follows brat annotation format. The formats are explained in the following.

  - Each row represents a annotation with opinion target or opinion expression or relation

  - For opinion target row, it consisits of three columns separated by `\t`.
  	- The first colum string starts with `T`
  	- The second colum string is `OpinionTarget left right`. The offsets in the plain text representation of the Amazon review are denoted by left and right.
  	- the third colum string  is the exact string representation of the opinion target.

  - For opinion expression row, it consisits of three columns separated by '\t'.
  	- The first colum string starts with `T`
  	- The second colum string is `OpinionExpression left right`. The offsets in the plain text representation of the Amazon review are denoted by left and right.
  	- the third colum string  is the exact string representation of the opinion expression.

  - For relation row, it consisits of two columns separated by '\t'.
  	- The first colum string starts with `R`
  	- The second colum string is `ExpressionTarget Arg1:T%d Arg2:T%d`. The Arg1 represents opinion expression and the Arg2 represents opinion target.
