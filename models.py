# Import standard libraries
from typing import List

# Import third-party libraries
from pydantic import BaseModel


class IndexRequest(BaseModel):
    use_case: str
    document_ids: List[str]
      
      
class Search(BaseModel):
    query: str
    num_results = 10
    context: int = 0
    
