from flask import Flask, render_template


# App config.
DEBUG = True
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("book_rec.html")


if __name__ == "__main__":
    app.run()