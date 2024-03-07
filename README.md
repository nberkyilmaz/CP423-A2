# Text Retrieval & Search Engine (CP423)

### Assignment 2 - Group 6 Submission

## Members:
- Gadd, Bryan

- Jain, Maheep

- Khamphavong, Osaka

- Simion, Alexandra

Due Date/Submission Date: July 9th, 2023, 11:59pm

## Summary:
The purpose of this assignment was to create a functioning program to searching up phrase queries in a dataset in Python and then ranking them by using TF-IDF. This involved preprocessing the data, implementing an positional index data structure, taking input from the user, implementing a phrase queries function, testing the code for errors, and running a set of queries that return data from the dataset.

## Requirments:
To run this project you will need to run the following command to install the required modules/libraries:

pip install -r .\requirements.txt

## Summary of files:
### Q1.
Contains the positional index data structure along with preprocessing function and phrase queries functions.
- The positional index structure allows the addition/updating of index's and can print the structure for debugging.
- The phrase query functions there are two helper functions, we use proximity_position to find positional in the same document, then merge algorithm to combine the docID into one list then give an output.

Assumption:
In phrase queries function we made an assumption that the user want to search for phrase that exist within the documents so the input would take a phrase and then the output would show in format: {docID: [position of the word]} (the position would be in an order because each word needs to be next to each other). If user enter phrase that does not exist in the document it will return empty result. 

### Q2.
Contains the TF, TF-IDF, and Cosine similarity functions. Q2 generates the matrix with its appropriate size and fills it with zeros. There's a function to calculate the TF-IDF numbers then populate the existing matrix with those numbers. The query vector is then generated in a separate function. To calculate the tf-idf score we need the tf-idf numbers from the matrix multiplied by the query vector generated (or more specifically the dot product between the two values). There's also another function that returns the top 5 most relevant documents based on the top 5 highest calculated tf-idf scores. The cosine_sim function is similar to the function that returns the top 5 most relevant documents based on tf-idf scores. The only difference is the top 5 values are based off the cosine similarity scores, instead. 

Assumption:
Log is always base 10 for any calculation. 

## How to run the program:
1. Refer to Requirements to install the required libraries.
2. Run Main.py.
3. You will be prompted to enter which function you want to run (or exit the program):
    - 0 = Exit
    - 1 = Phrase query
    - 2 = TF-IDF
    - 3 = Cosine Similatiry
    - Enter a number and press enter to select the function (or exit the program).
4. After picking the function you will be prompted to enter the query. Enter the query and press enter.
5. At this point phrase query will process and output the result. If TF-IDF or Cosine Similatiry was picked then you will be asked to pick which weight scheme to use.
    - 1 = Binary
    - 2 = Raw Count
    - 3 = Term Frequency (NOTE: It will take a few minutes or longer to process this scheme based on the vocabulary size.)
    - 4 = Log Normalization
    - 5 = Double Normalization (NOTE: It will take a few minutes or longer to process this scheme based on the vocabulary size.)
    - Enter a number and press enter to select the weight schemna. Once processed the results are outputed.
6. Once the selected function outputs its results you will return to step 3 where you can pick another function to run or exit the program.

## Program Example:
### Phrase Query
Options:
0 = Exit program

1 = Phrase Query

2 = TD-IDF

3 = Cosine

Enter an option: 1
Enter the query: sherlock holmes

Creating Positional Index for the first time...
Finished creating Positional Index.

Documents that contain the phrase

Read as { DocId: [Positions of the phrase], ...}

{3: [7664, 7665], 8: [8081, 8082], 53: [1451, 1452], 86: [10331, 10332], 131: [6473, 6474], 206: [10004, 10005]}

### TF-IDF
Options:
0 = Exit program

1 = Phrase Query

2 = TD-IDF

3 = Cosine

Enter an option: 2

Enter the query: sherlock holmes

Pick an option for the TF weight scheme:

1 = Binary

2 = Raw count

3 = Term Frequency

4 = Log Normalization

5 = Double Normalization

Enter an option: 1

Creating Positional Index for the first time...
Finished creating Positional Index.

TF-IDF Result:

Top 5 dopcumets are:
Document 189
Document 213
Document 45
Document 218
Document 102

### Cosine Similarity
Options:
0 = Exit program

1 = Phrase Query

2 = TD-IDF

3 = Cosine

Enter an option: 3

Enter the query: sherlock holmes

Pick an option for the TF weight scheme:

1 = Binary

2 = Raw count

3 = Term Frequency

4 = Log Normalization

5 = Double Normalization

Enter an option: 1

Creating Positional Index for the first time...
Finished creating Positional Index.

Cosine Similarity Result:

Top 5 dopcumets are:
Document 238
Document 5
Document 3
Document 201
Document 170

## Cosine Similarity Report:
NOTE/ASSUMPTION: We based the pros and cons on the performance of the scheme as there was no metric mentioned in the assignment outline.

We used the first 48 files from the dataset for this report. The query used was 'sherlock holmes'.

### Binary:
Pros:
- Quick to process (simple calculation)

Cons:
- Didn't notice any cons.

Output:
TD-IDF Matrix
[[1.20411998 0.77815125 0.98227123 ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.98227123 ... 0.         0.         0.        ]
 [0.         0.77815125 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 1.38021124 1.38021124 1.38021124]]

Cosine Similarity Result:
Top 5 dopcumets are:
Document 5
Document 3
Document 10
Document 14
Document 41

### Raw Count
Pros:
- Quick to process (simple calculation)

Cons:
- Didn't notice any cons.

Output:
TD-IDF Matrix
[[1.20411998 0.77815125 0.98227123 ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.98227123 ... 0.         0.         0.        ]
 [0.         0.77815125 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 1.38021124 1.38021124 1.38021124]]

Cosine Similarity Result:
Top 5 dopcumets are:
Document 5
Document 3
Document 10
Document 14
Document 41

### Term Frequency
Pros:
- Didn't notice any pros.

Cons:
- Process heavy (complex calculation that takes a while)

Output:

TD-IDF Matrix
[[1.20411998 0.77815125 0.98227123 ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.98227123 ... 0.         0.         0.        ]
 [0.         0.77815125 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 1.38021124 1.38021124 1.38021124]]

Cosine Similarity Result:
Top 5 dopcumets are:
Document 5
Document 3
Document 10
Document 14
Document 41

### Log Normalization
Pros:
- Quick to process (simple calculation)
- Normalized values

Cons:
- Didn't notice any cons.

Output:
TD-IDF Matrix
[[0.36247623 0.23424687 0.29569311 ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.29569311 ... 0.         0.         0.        ]
 [0.         0.23424687 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.41548498 0.41548498 0.41548498]]

Cosine Similarity Result:
Top 5 dopcumets are:
Document 5
Document 3
Document 10
Document 14
Document 41

### Double Normalization
Pros:
- Didn't notice any pros.

Cons:
- Process heavy (complex calculation that takes a while)

Output:
[[1.20411998 0.77815125 0.98227123 ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.98227123 ... 0.         0.         0.        ]
 [0.         0.77815125 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 1.38021124 1.38021124 1.38021124]]
