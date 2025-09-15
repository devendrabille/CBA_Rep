from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    response = ""
    if request.method == 'POST':
        user_input = request.form['user_input'].lower()

        if "hi" in user_input:
            response = "Hi, how can I help you?"
        elif "4500000001" in user_input:
            response = "PO Status: Active. Vendor: USSU-VSF04. Items: BKR-300 Frame, Handle Bars, Seat. Delivery: 10/08/2017."
        elif "4500122023" in user_input:
            response = "No PO found for this order number."
        else:
            response = "Sorry, I couldn't process that right now. Please try again later."

    return '''
        <h2>Ask your SAP assistant</h2>
        <form method="post">
            <input name="user_input" type="text">
            <input type="submit" value="Ask">
        </form>
        <p><strong>Bot Response:</strong> {}</p>
    '''.format(response)

if __name__ == '__main__':
    app.run(debug=True)