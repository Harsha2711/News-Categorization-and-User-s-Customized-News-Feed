from flask import Flask, request, jsonify 
from flask_cors import CORS  # Import CORS from flask_cors 
from lstwsvm_multiclass import LSTWSVM_MultiClass
from collections import Counter
import pickle 
 
app = Flask(__name__) 
CORS(app)  # Enable CORS for all routes 
 
# Load the trained model and vectorizer 
with open('../ML/trained_model.pkl', 'rb') as model_file: 
    lstwsvm_model = pickle.load(model_file) 
 
with open('../ML/vectorizer.pkl', 'rb') as vectorizer_file: 
    vectorizer = pickle.load(vectorizer_file)
 
test_text = ["Test news headline"]
test_vector = vectorizer.transform(test_text)
print("TF-IDF transformation successful!") 

@app.route('/categorize_news', methods=['POST']) 
def categorize_news(): 
    data = request.get_json() 
    news_data = data['newsData'] 
 
    # Vectorize the titles 
    titles = [article['title'] for article in news_data] 
    titles_tfidf = vectorizer.transform(titles) 
 
    # Predict categories 
    predicted_categories = lstwsvm_model.predict(titles_tfidf) 
 
    # Create a dictionary to store categorized news 
    categorized_news = {} 
    for article, category in zip(news_data, predicted_categories): 
        # Include all attributes for each article 
        article.update({'category': category}) 
        if category not in categorized_news: 
            categorized_news[category] = [] 
        categorized_news[category].append(article) 
 
    # Add CORS headers to the response 
    response = jsonify(categorized_news) 
    response.headers.add('Access-Control-Allow-Origin', '*') 
    response.headers.add('Access-Control-Allow-Methods', 'POST') 
 
    return response 
@app.route('/customized_news', methods=['POST']) 
def customized_news(): 
    data = request.get_json() 
    categorized_news = data['categorizedNews'] 
    recentlyviewed_news=data['recentlyViewedArticles']
    customized_news=[]
    categories = [article['category'] for article in recentlyviewed_news] 
    category_counter = Counter(categories)
    sorted_categories = sorted(category_counter.items(), key=lambda x: x[1], reverse=True)
    sorted_category_list = [category for category, _ in sorted_categories]
    for i in range(0,3):
        if len(sorted_category_list)>i:
            for article in categorized_news[sorted_category_list[i]]:
                if(article['title'] not in recentlyviewed_news):
                    customized_news.append(article)
    # Add CORS headers to the response 
    response = jsonify(customized_news) 
    response.headers.add('Access-Control-Allow-Origin', '*') 
    response.headers.add('Access-Control-Allow-Methods', 'POST') 
    return response 
if __name__ == '__main__': 
    app.run(debug=True) 