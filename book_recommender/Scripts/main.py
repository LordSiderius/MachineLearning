from flask import Flask, request, render_template
import book_rec_class


DEBUG = True

book_database = book_rec_class.Book_database()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/results", methods=['POST'])
def results():
    
    book_title = request.form.get('BookTitle')
    book_author = request.form.get('BookAuthor')
    
    try:
        results = book_database.recommend(book_title, book_author)
    except:
        return render_template('error.html')
    
    return render_template('results.html', book_title=book_title, book_author=book_author, results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
