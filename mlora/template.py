shipinhao_template = f"""Please classify the following text, 0 indicates normal text, 1 indicates non-compliant text: 
Title: {{title}}, ParentComment: {{parentComment}}, SubComment{{subComment}}
Result: 
"""

xiaoshijie_template = f"""Please classify the following text, 0 indicates normal text, 1 indicates non-compliant text: 
Title: {{title}}, ParentComment: {{parentComment}}, SubComment{{subComment}}
Result: 
"""

gongzhongpinglun_template = f"""Please classify the following text, 0 indicates normal text, 1 indicates non-compliant text: 
Title: {{title}}, Content: {{content}}
Result: 
"""

agnews_template = f"""Please categorize the following news text and choose the most appropriate answer from A, B, C, D, then output the letter of the answer directly.
{{sentence}}
A. World
B. Sports
C. Business
D. Science or Technology
Answer: """

amazon_template = f"""Please perform sentiment analysis on the following review data and choose the most appropriate answer from A, B, C, D, E, then output the letter of the answer directly.
{{sentence}}
A. very negative
B. negative
C. neutral
D. positive
E. very positive
Answer: """

dbpedia_template = f"""Please identify the entities mentioned in the following text and choose the most appropriate answer from A, B, C, D, E, F, G, H, I, J, K, L, M, N, then output the letter of the answer directly.
{{sentence}}
A. Company
B. Educational Institution
C. Artist
D. Athlete
E. Office Holder
F. Mean of Transportation, 
G. Building
H. Natural Place
I. Village
J. Animal
K. Plant
L. Album
M. Film
N. Written Work
Answer: """

yahoo_template = f"""Please classify the following text and choose the most appropriate answer from A, B, C, D, E, F, G, H, I, J, then output the letter of the answer directly.
{{sentence}}
A. Society & Culture
B. Science & Mathematics
C. Health
D. Education & Reference
E. Computers & Internet
F. Sports
G. Business & Finance
H. Entertainment & Music
I. Family & Relationships
J. Politics & Government
Answer: """
