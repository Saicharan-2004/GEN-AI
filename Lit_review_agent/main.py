from fastapi import FastAPI
import topic


app = FastAPI()

app.include_router(topic.router)

 