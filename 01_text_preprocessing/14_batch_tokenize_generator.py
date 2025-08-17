def read_lines(filename):


    with open(filename, 'r', encoding='utf-8')as file:
        for line in file:
            yield line.strip()



if __name__ == "__main__":

    sample_text = """Hello World
This is a test
Generators save memory
We process one line at a time"""


    with open("sample.txt", "w"):
        f.write(sample_text)

    

    print("Ready lines one by one")
    for line in read_lines("sample_text"):
        print(f"Line '{line}'")






