# PDF dataExtractproject
project for extracting content from pdfs/ images. 
Here using OCR for extracting content from images

## Development
### To set up development environment
Make sure you have docker installed <br>
<br>


- Clone the repo
- cd to repository folder

#### run below command to build image, based on Dockerfile pulled from repo 
```  
$ docker image build -t ocr_pdf_model . 
```

#### once image is build, then run following command to run container 
```
$ docker run -d --name ocr_pdf_model_con ocr_pdf_model
``` 

#### access the container using below comment
```  
$ docker exec -it ocr_pdf_model_con bash 
```
