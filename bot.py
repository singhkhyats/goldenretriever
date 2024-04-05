import nltk
import numpy as np
from newspaper import Article
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt')
if not nltk.data.find('corpora/stopwords'):
    nltk.download('stopwords')

print("Setting up the AI...")

urls = [
    'https://en.wikipedia.org/wiki/Applications_of_artificial_intelligence',
    'https://en.wikipedia.org/wiki/Virtual_assistant',
    'https://phys.org/news/2021-04-ai-gauge-emotional-state-cows.html',
    'https://en.wikipedia.org/wiki/Artificial_intelligence_arms_race',
    'https://en.wikipedia.org/wiki/AlphaFold#AlphaFold_2,_2020',
    'https://www.theverge.com/2022/12/7/23497980/elliq-companion-robot-2-0-elderly-care-features-conversation-prompts',
    'https://en.wikipedia.org/wiki/AARON'
]

extraGhostData = "In his 1988 book Mind Children, roboticist Hans Moravec proposed that a future supercomputer might be able to resurrect long-dead minds from the information that still survived. For example, such can include information in the form of memories, filmstrips, social media interactions, modeled personality traits, personal favourite things, personal notes and tasks , medical records, and genetic information Ray Kurzweil, American inventor and futurist, believes that when his concept of singularity comes to pass, it will be possible to resurrect the dead by digital recreation.Such is one approach in the concept of digital immortality, which could be described as resurrecting deceased as digital ghosts or digital avatars. In the context of knowledge management, virtual persona could aid in knowledge capture, retention, distribution, access and use and continue to learn. Issues include post-mortem privacy, and potential use of personalised digital twins and associated systems by  big data firms and advertisers. Related alternative approaches of digital immortality include gradually replacing neurons in the brain with advanced medical technology (such as  nanobiotechnology as a form of mind uploading"

#Initialize article objects and download and parse each one
corpus = ''
articles = []
for url in urls:
    article = Article(url)
    article.download()
    article.parse()
    articles.append(article)
    corpus += article.text + '\n\n'

corpus += extraGhostData

# Tokenize the corpus into sentences and concatenate to fit the model's token limit
sentence_list = nltk.sent_tokenize(corpus)
optimized_corpus = ' '.join(sentence_list[:200])  

chunks = [' '.join(sentence_list[i:i+5]) for i in range(0, len(sentence_list), 5)]

# Load the question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def find_relevant_context(question, chunks, top_n):
    # Include the question as part of the documents to compute TF-IDF
    documents = [question] + chunks
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # The first row corresponds to the question's TF-IDF
    question_tfidf = tfidf_matrix[0]

    # Compute similarity scores between the question and each chunk
    similarity_scores = (tfidf_matrix * question_tfidf.T).toarray()[1:]  # Exclude the first row (question itself)

    # Rank chunks by their similarity scores
    ranked_chunks = sorted([(score, chunk) for score, chunk in zip(similarity_scores, chunks)], reverse=True)

    # Return the top N chunks based on their scores
    top_chunks = [chunk for score, chunk in ranked_chunks[:top_n]]
    return top_chunks

def combined_bot_response(user_input, qa_pipeline, chunks, sentence_list):
    # First, try to get an answer using the more sophisticated TF-IDF and QA pipeline method
    relevant_chunks = find_relevant_context(user_input, chunks, top_n=5)
    answers = []

    for chunk in relevant_chunks:
        result = qa_pipeline(question=user_input, context=chunk)
        if result['score'] > 0.1:  # Adjust the threshold as necessary
            answers.append((result['score'], result['answer']))

    if answers:
        answers.sort(reverse=True)
        top_answer = answers[0][1]
        if top_answer and len(top_answer.split()) > 3:  # Check if the answer is sufficiently detailed
            return top_answer

    # If the first method fails to return a satisfactory answer, fallback to the keyword-based approach
    user_input = user_input.lower()
    sentences = sentence_list + [user_input]
    cm = CountVectorizer().fit_transform(sentences)
    similarity_scores = cosine_similarity(cm[-1], cm)
    similarity_scores_list = similarity_scores.flatten()
    indices = np.argsort(similarity_scores_list)[-3:-1]  # Get top 2 indices, excluding the user's input itself

    fallback_response = ''
    for index in reversed(indices):
        if similarity_scores_list[index] > 0.0:
            fallback_response += ' ' + sentences[index]

    return fallback_response if fallback_response else "I am sorry, I do not understand or cannot find an answer."


print("Hi I'm Golden Retriever and I'm ready to retrieve! Type in your question.")

while True:
    user_input = input("\nSay something: ").lower()
    if user_input == 'quit':
        print("Exiting AI chat. Goodbye!")
        break
    print("Golden Retriever:", combined_bot_response(user_input, qa_pipeline, chunks, sentence_list))
