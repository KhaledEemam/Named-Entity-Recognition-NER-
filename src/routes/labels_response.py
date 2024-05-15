from fastapi import FastAPI , APIRouter , status
from fastapi.responses import JSONResponse
from controllers import DataController
from tasks import get_tags

response_route = APIRouter()

@response_route.get('/get-labels')
def get_labels(sentence : str) :
    is_valid , response_message = DataController().validate_sentence_length(sentence)

    if not is_valid :
        return JSONResponse(
            status_code = status.HTTP_400_BAD_REQUEST ,
            content = {"Valid" : is_valid , "Message" : response_message}
        )
    
    output_label = get_tags(sentence)
    
    if is_valid :
        return JSONResponse(
            content = {"Valid" : is_valid , "Message" : output_label}
        )