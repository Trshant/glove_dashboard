import re
import sys
import csv

from app.core.elastic import elastic_search
from app.core.glove import glove

from app.libs.utils import util


## this bit od code is simple to make 
## sure we dont get the overflow or other error
## https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


INPUT_PATH="../case_input/"

def read_csv_and_yield_one_line( filepath , input_column):    
    with open( filepath , encoding="utf8") as f:
        csv_reader = csv.DictReader(f)
        # skip the header
        next(csv_reader)
        # show the data
        for line in csv_reader:
            yield line[input_column]


def read_txt_and_yield_one_line( filepath ):    
  """
  Yields each line in the specified file.

  Args:
      filepath (str): The path to the file to read.

  Yields:
      str: Each line from the file.
  """
  try:
    with open(filepath, 'r') as f:
      for line in f:
        yield line.rstrip('\n')  # Remove trailing newline character
  except FileNotFoundError:
    print(f"Error: File not found - {filepath}")


def read_create_sentences(file_path):
    #####
     
    #####
    sentences = []
    word_list_total = []
    lines = read_txt_and_yield_one_line(file_path)
    for line in lines:
        sentences.append(line)
    return sentences

# Usage example:
if __name__ == "__main__":
    case_name = "IMDB"
    input_file_name = "IMDB.csv"
    input_column = "review"
    file_path = INPUT_PATH+case_name+"/"+input_file_name
    sentences = read_csv_and_yield_one_line( file_path , input_column )
    module_glove = glove(case_name)
    ## module_elastic = elastic_search( case_name.lower() )
    module_glove.add_sentences(sentences)
    #module_elastic.insert_data(sentences)
    module_glove.save_model()

    









    