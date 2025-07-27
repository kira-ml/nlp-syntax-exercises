import re
from html.parser import HTMLParser



def get_sample_html():


    return """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Welcome to <b>my site</b></h1>
            <p>This is a <a href='https://example.com'>link</a>.</p>
            <div class="footer">Footer content here</div>
        </body>
    </html>

"""


def remove_html_with_regex(text):


    clean_text = re.sub(r'<[^>]+>', '', text)

    return clean_text


class HTMLStripper(HTMLParser):

    def __init__ (self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)
    
    def get_data(self):
        return ''.join(self.fed)
    


def remove_html_with_parser(text):

    stripper = HTMLStripper()
    stripper.feed(text)
    return stripper.get_data()



def compare_methods():

    html_text = get_sample_html()

    print("Original HTML:")
    print(html_text)


    print("\n Cleaned with REGEX:")
    print(remove_html_with_regex(html_text))

    print("\n Cleaned with HTMLParser:")
    print(remove_html_with_parser(html_text))



if __name__ == "__main__":
    compare_methods()