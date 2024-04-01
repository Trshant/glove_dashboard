from fastapi import FastAPI, APIRouter, Query, HTTPException, Request
from fastapi.templating import Jinja2Templates

from typing import Optional, Any
from pathlib import Path

from app.schemas import RecipeSearchResults, Recipe, RecipeCreate
from app.recipe_data import RECIPES

from app.core.elastic import elastic_search
from app.core.glove import glove

from app.libs.utils import util

import json

from urllib.parse import parse_qs

BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))

CASE_NAME = "IMDB"

app = FastAPI(title="Recipe API", openapi_url="/openapi.json")

api_router = APIRouter()

module_elastic = elastic_search( CASE_NAME.lower() )
module_glove  = glove( CASE_NAME )

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
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request, "recipes": RECIPES},
    )


@api_router.post("/submit-form", status_code=200)
async def glove_elastic(request: Request) -> dict:
    """
    Root GET
    """

    raw_body = await request.body()
    decoded_body = raw_body.decode('utf-8')
    parsed_data = parse_qs(decoded_body)
    print( parsed_data["name"] )


    words = parsed_data["name"]
    count=15
    neg_words = [] ## ["australia","scandinavia","branch"]
    filter_negatives = True

    data_to_present = module_glove.get_similar_words(words,count,neg_words,filter_negatives)

    form_terms = TEMPLATES.TemplateResponse(
        "form_term.html",
        {"request": request, "data":data_to_present},
    )
    return form_terms

@api_router.post("/submit-search", status_code=200)
async def elastic(request: Request):
    """
    Root GET
    """
    def get_data(item):
        data = {}
        data["id"] = item["_id"]
        data["content"] = item["_source"]["content"]
        return data

    raw_body = await request.body()
    decoded_body = raw_body.decode('utf-8')
    parsed_data = parse_qs(decoded_body)
    
    pos_terms = parsed_data.get("pro[]" , [])
    neg_terms = parsed_data.get("con[]" , [])

    elastic_query = util.list2elastic( pos_terms , neg_terms )
    print(elastic_query)

    return_result = module_elastic.search_index( elastic_query )

    data_to_return = {}
    data_to_return["total"] = return_result["hits"]["total"]["value"]
    data_to_return["data"] = list( map( get_data , return_result["hits"]["hits"] ) )

    form_terms = TEMPLATES.TemplateResponse(
        "search_results.html",
        {
            "request": request, 
            "data":data_to_return["data"],
            "total":data_to_return["total"],
            "query": json.dumps( elastic_query , indent=4),
        },
    )
    return form_terms


app.include_router(api_router)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
