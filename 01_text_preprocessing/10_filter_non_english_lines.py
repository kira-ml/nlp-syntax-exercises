import unicodedata

def filter_english_lines(lines):


    english_lines = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if is_probably_english(cleaned):
            english_lines.append(cleaned)
    
    return english_lines



def is_probably_english(text):

    latin_count = 0
    total_letters = 0


    for char in text:
        if char.isalpha():
            total_letters += 1

            try:
                if "LATIN" in unicodedata.name(char):
                    latin_count += 1

            except ValueError:
                continue


    if total_letters == 0:
        return False
    

    return latin_count / total_letters > 0.8