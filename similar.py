from similarity_docs import get_similarity

f1 = open("article1", "r")
f2 = open("article2", "r")
text1 = f1.read()
text2 = f2.read()
print(get_similarity(text1, text2))
f1.close()
f2.close()