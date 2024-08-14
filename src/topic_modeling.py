from gensim.corpora import Dictionary
from gensim.models import LdaModel


def perform_topic_modeling(documents, num_topics=5):
    texts = [doc.page_content.split() for doc in documents]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    
    for i, doc in enumerate(documents):
        doc_topic_dist = lda.get_document_topics(corpus[i])
        dominant_topic = max(doc_topic_dist, key=lambda x: x[1])[0]
        doc.metadata['topic'] = dominant_topic
    
    return documents
