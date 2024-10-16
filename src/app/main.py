from fastapi import FastAPI, APIRouter, Query, HTTPException, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from typing import Optional, Any, Annotated
from pathlib import Path

from app.schemas import RecipeSearchResults, Recipe, RecipeCreate

from app.core.elastic import elastic_search
from app.core.glove import glove
from app.core.files_processer import FileProcessor

from app.libs.utils import util
from app.libs.log_manager import *

import json

from urllib.parse import parse_qs

BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))

CASE_NAME = "IMDB"

app = FastAPI(title="Recipe API", openapi_url="/openapi.json")

app.mount("/static", StaticFiles(directory="./app/static"), name="static")

api_router = APIRouter()

class ApplicationMain:
    def __init__(self):
        self.module_elastic =  None # elastic_search( CASE_NAME.lower() )
        self.module_glove  = None # glove( CASE_NAME )
        self.logger = get_logger("application")
    def select_docuSet(self, docuset_name):
        self.module_elastic =  elastic_search( docuset_name.lower() )
        self.module_glove  = glove( docuset_name )
        self.processor = FileProcessor()

application = ApplicationMain()
application.select_docuSet(CASE_NAME)

"""
urls

index -> template
{ word(positive) , words(negative) }    -> return from the model
                                        -> list of results from the search

"""

# Updated to serve a Jinja2 template
# https://www.starlette.io/templates/
# https://jinja.palletsprojects.com/en/3.0.x/templates/#synopsis


@api_router.get("/", status_code=200)
def root(request: Request) -> dict:
    """
    Root GET
    """
    application.logger.info("starting the system!")
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request},
    )


@api_router.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file) }

async def write_file(name, fp ):
    f = open("./app/static/uploads/"+name, "xb")
    data = await fp.read()
    f.write(data)
    f.close()

@api_router.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    await write_file( file.filename, file )
    text, num_pages = application.processor.process_file( "./app/static/uploads/" + file.filename )
    application.module_elastic.insert_data(text)
    return {"filename": file.filename , "size":file.size , "headers": file.headers , "content_type": file.content_type }

@api_router.post("/analyze_terms", status_code=200)
async def glove_elastic(request: Request) -> dict:
    """
    Root GET
    """

    raw_body = await request.body()
    decoded_body = raw_body.decode('utf-8')
    parsed_data = parse_qs(decoded_body)

    status, form_elements = util.get_terms( parsed_data["open"][0] )
    application.logger.debug(form_elements)
    status , words = util.get_terms( parsed_data["name"][0] )
    application.logger.debug(words)
    count_terms = 15
    data_to_present = glove_fe(pos_terms=words)
    count, documents , query = elastic_search_fe( words , [] )


    form_terms = TEMPLATES.TemplateResponse(
        "form_term.html",
        {
            "request": request,
            "terms":data_to_present ,
            "pos_terms":words,
            "documents":documents,
            "count":count_terms,
            "total":count,
            "query": json.dumps( query , indent=4) ,
            "form_elements": form_elements
         },
    )
    application.logger.info("starting the system!")
    return form_terms

def glove_fe( pos_terms:list=[],count:int=15,neg_terms:list=[],filter_negatives:bool=True ):
    return_data = []
    list_to_check = []
    return_data += [ (term , 100) for term in pos_terms ]
    list_to_check += pos_terms + neg_terms
    countdown = count - ( len(pos_terms) + len( neg_terms ) )
    similar_words = application.module_glove.get_similar_words(pos_terms, 200 ,neg_terms,filter_negatives)
    similar_words.reverse()
    while countdown > 0:
        popped_similar_word = similar_words.pop()
        if popped_similar_word[0] not in list_to_check :
            return_data.append( popped_similar_word )
            list_to_check.append( popped_similar_word[0] )
            countdown -= 1

    return return_data


def elastic_search_fe( pos_terms , neg_terms ):

    def get_data(item):
        data = {}
        data["id"] = item["_id"]
        data["content"] = item["_source"]["content"]
        return data

    elastic_query = util.list2elastic( pos_terms , neg_terms )
    ##print(elastic_query)

    return_result = application.module_elastic.search_index( elastic_query )
    print( return_result )
    data_to_return = {}
    total_count = return_result["hits"]["total"]["value"]
    data = list( map( get_data , return_result["hits"]["hits"] ) )
    return total_count , data , elastic_query


@api_router.post("/submit-glove", status_code=200)
async def elastic(request: Request):
    """
    Root GET
    """


    raw_body = await request.body()
    decoded_body = raw_body.decode('utf-8')
    parsed_data = parse_qs(decoded_body)

    status, form_elements = util.get_terms( parsed_data["open"][0] )
    application.logger.debug(form_elements)

    pos_terms = parsed_data.get("pro[]" , [])
    neg_terms = parsed_data.get("con[]" , [])
    count_terms = int( parsed_data.get("count" , 0)[0])

    data_to_present = glove_fe(pos_terms=pos_terms,count=count_terms, neg_terms=neg_terms)

    count, documents , query = elastic_search_fe( pos_terms , neg_terms )

    form_terms = TEMPLATES.TemplateResponse(
        "form_term.html",
        {
            "request": request,
            "terms":data_to_present ,
            "pos_terms":pos_terms,
            "documents":documents,
            "count":count_terms,
            "total":count,
            "query": json.dumps( query , indent=4) ,
            "form_elements": form_elements
         },
    )
    return form_terms



@api_router.post("/submit-search", status_code=200)
async def elastic(request: Request):
    """
    Root GET
    """


    raw_body = await request.body()
    decoded_body = raw_body.decode('utf-8')
    parsed_data = parse_qs(decoded_body)

    status, form_elements = util.get_terms( parsed_data["open"][0] )
    application.logger.debug(form_elements)

    pos_terms = parsed_data.get("pro[]" , [])
    neg_terms = parsed_data.get("con[]" , [])
    count_terms = int( parsed_data.get("count" , 0)[0])

    data_to_present = glove_fe(pos_terms=pos_terms,count=count_terms, neg_terms=neg_terms)

    count, documents , query = elastic_search_fe( pos_terms , neg_terms )

    form_terms = TEMPLATES.TemplateResponse(
        "search_results.html",
        {
            "request": request,
            "terms":data_to_present ,
            "pos_terms":pos_terms,
            "documents":documents,
            "count":count_terms,
            "total":count,
            "query": json.dumps( query , indent=4) ,
            "form_elements": form_elements
         },
    )
    return form_terms


app.include_router(api_router)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
