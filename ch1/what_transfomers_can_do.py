# from transformers import pipeline
#
# classifier = pipeline('sentiment-analysis')
# results = classifier("I've been waiting for a HuggingFace course my whole life")
#
# print(results)
#
# # Passing several sentences
# results = classifier(
#     ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
# )
# print(results)
#
# # Zero-shot classification
# classifier = pipeline('zero-shot-classification')
# results  = classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"],
# )
#
# print(results)
#
# # Text generation
# generator = pipeline('text-generation')
# results = generator(
#     "In this course, we will teach you how to",
#     num_return_sequences=5, # Specifies the number of different sentences generated
#     max_length=30 # the total length of the output text
# )
# print(results)
#
# generator = pipeline("text-generation", model="distilgpt2")
# results = generator(
#     "In this course, we will teach you how to",
#     max_length=30,
#     num_return_sequences=2,
# )
# print(results)
#
# # Mask filling
# unmasker = pipeline("fill-mask")
# results = unmasker("This course will teach you all about <mask> models.", top_k=2)
# print(results)
#
# # Named entity recognition
# ner = pipeline('ner', grouped_entities=True)
# results = ner("My name is Sean Huvaya and I work at Impressions in Queens")
# print(results)
#
# # Question answering
# question_answerer = pipeline("question-answering")
# results = question_answerer(
#     question="Where do I work?",
#     context="My name is Sean Huvaya and I work at Impressions in Queens",
# )
# print(results)
#
# # Summarization
# summarizer = pipeline("summarization")
# results = summarizer("""
# ## Transformers
#
# - Provides APIs and tools to easily download and train state-of-the-art pre-trained models
# - Advantages of using pre-trained models:
#     1. Can reduce computational costs
#     2. Can reduce carbon footprint
#     3. Saves the time and resources required to train a model from scratch
# - These models support tasks such as:
#     1. **Natural Language Processing:** text classification, named entity recognition, question answering, language modeling, code generation, summarization, translation, multiple choice and text generation
#     2. **Computer Vision:** image classification, object detection and segmentation
#     3. **Audio:** automatic speech recognition and audio classification
#     4. **Multimodal:** table question answering, optical character recognition, information extraction from scanned documents, vidoe classification and visual question answering
# - The `pipeline()` is the easiest and fastest way to use a pre-trained model for inference
# - It connects a model with its necessary preprocessing and post-processing steps, allowing for the direct input of any text and get an intelligible answer
# - Steps involved when passing some text to a pipeline:
#     1. The text is preprocessed into a format the model can understand
#     2. The preprocessed inputs are passed to the model
#     3. The predictions of the model are post-processed, so you can make sense of them
# """)
# print(results)
#
# # Translation
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
# results = translator("Ce cours est produit par Hugging Face.")
# print(results)
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformers library")